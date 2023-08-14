import pandas as pd
import argparse
import json
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from redis import StrictRedis
from DataSource import ENRICHED_DOMAIN
from common import define_logger
from DatabaseConnector import DatabaseConnector, COMMIT_BATCH_SIZE
from Model.MarkovModel import MarkovModel
from Model.SnaModel import SnaModel
from Model.MLModel import MLModel
from Model.AggragatorModel import AggragatorModel

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
logger = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help='train new models')
    parser.add_argument("--listen", action='store_true', help='listen for new enriched domains')
    parser.add_argument("--logger", type=str, default='model_manager', help='logger name')
    parser.add_argument("--pkl-path", type=str, default='10K10K.pkl', help='path to pickle to save/load the model file')
    parser.add_argument("--limit", type=int, default=1000, help='limit records per classification')
    parser.add_argument("--retrain", type=int, default=0, help='model retraining every X hours, 0 for no retraining')
    parser.add_argument("--postgresql-host", type=str, default='localhost', help="postgresql host")
    parser.add_argument("--postgresql-port", type=int, default=5432, help="postgresql port")
    parser.add_argument("--postgresql-username", type=str, help="postgresql username")
    parser.add_argument("--postgresql-password", type=str, help="postgresql password")
    parser.add_argument("--debug-level", type=str, default='INFO', help='logging debug level')
    args = parser.parse_args()
    print(args)
    global logger 
    logger = define_logger(args.logger, args.debug_level)
    db_conn = DatabaseConnector(
        args.postgresql_username,
        args.postgresql_password,
        args.postgresql_host,
        args.postgresql_port,
        logger
    )
    if args.train:
        train(db_conn, args.pkl_path, args.limit)
    if args.listen:
        listen(db_conn, args.pkl_path, args.limit, args.retrain)

def train(db_conn, pkl_path, limit=10000):
    global logger
    if logger is None:
        logger = define_logger('model_manager')
    X_benign = db_conn.get_records(label=0, limit=limit, hosting=False, columns=None)
    X_malicious = db_conn.get_records(label=1, limit=limit, hosting=False, columns=None)
    X = pd.concat([X_malicious, X_benign])
    y = X['label']
    logger.info('X_malicious: %d, X_benign: %d, X: %d', len(X_malicious), len(X_benign), len(X))

    xgb_ml_model = MLModel(XGBClassifier(), logger=logger)
    lr_ml_model = MLModel(LogisticRegression(solver='lbfgs'), logger=logger)
    sna_model = SnaModel(logger=logger)
    markov_model = MarkovModel(logger=logger)
    model_list = [lr_ml_model, xgb_ml_model, markov_model, sna_model]
    agg_model = AggragatorModel(model_list, logger=logger)
    agg_model.train(X, y)
    agg_model.save_model(pkl_path)
    return agg_model

def listen(db_conn, pkl_path, limit, retrain):
    agg_model = AggragatorModel.load_model(pkl_path)
    redis = StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    predict_batch = []
    pubsub = redis.pubsub()
    pubsub.subscribe(ENRICHED_DOMAIN)
    logger.info('going into listening mode...')              
    for message in pubsub.listen():
        try:
            if message['type'] == 'message':
                domain = message['data']
                if db_conn.is_domain_record_exist(domain):
                    logger.info('%s already exists in the DB', domain)
                    continue
                features_str = redis.get(domain)
                features_dict = json.loads(features_str)
                logger.debug('features_dict %s', features_dict)
                X = pd.DataFrame.from_dict([features_dict])
                if retrain > 0 and datetime.now() - timedelta(hours=retrain) > agg_model.train_date:
                    logger.debug('more then %d hours past since last train of the model', retrain)
                    agg_model = train(db_conn, pkl_path, limit)
                else:
                    logger.debug('less then %d hours past since last train of the model', retrain)
                predict_df = agg_model.predict(X)
                predict_dict = predict_df.to_dict(orient='records')[0]
                logger.debug('predict_dict %s', predict_dict)
                join_dict = {**features_dict, **predict_dict}
                logger.debug('join_dict %s', join_dict)
                redis.set(domain, json.dumps(join_dict), ex=14400)
                logger.debug(predict_df)
                predict_batch.append(domain)
                logger.info('len(predict_batch): %d', len(predict_batch))
                if len(predict_batch) >= COMMIT_BATCH_SIZE:
                    logger.info('adding domains to DB')
                    for domain in predict_batch:
                        db_conn.add_domain_to_db(domain)
                    predict_batch.clear()
        except Exception:
            logger.exception(message)

if __name__ == "__main__":
    main()
