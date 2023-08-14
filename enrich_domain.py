import json
from redis import StrictRedis
from common import get_domain_from_url, define_logger
from DataSource import NEW_URL_TOPIC, ENRICHED_DOMAIN
from FeatureExtraction import feature_extractor
from DatabaseConnector import DatabaseConnector
import argparse

LOGGER_NAME = 'enrich_domain'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postgresql-host", type=str, default='localhost', help="postgresql host")
    parser.add_argument("--postgresql-port", type=int, default=5432, help="postgresql port")
    parser.add_argument("--postgresql-username", type=str, help="postgresql username")
    parser.add_argument("--postgresql-password", type=str, help="postgresql password")
    parser.add_argument("--debug-level", type=str, default='INFO', help='logging debug level')
    args = parser.parse_args()
    logger = define_logger(LOGGER_NAME, args.debug_level)
    logger.debug('args: %s', args)
    redis = StrictRedis(host='localhost', port=6379, db=0)
    db_conn = DatabaseConnector(
        args.postgresql_username,
        args.postgresql_password,
        args.postgresql_host,
        args.postgresql_port,
        logger
    )
    pubsub = redis.pubsub()
    pubsub.subscribe(NEW_URL_TOPIC)
    logger.info('going into listening mode...')
    for message in pubsub.listen():
        try:
            if message['type'] == 'message':
                message_content = json.loads(message['data'])
                url = message_content['url'] #.decode('utf-8')
                label = message_content['label']
                domain = get_domain_from_url(url)
                if domain and not redis.get(domain):
                    if db_conn.is_domain_record_exist(domain):
                        logger.debug('%s domain record already exists', domain)
                        continue
                    features = feature_extractor.extract_domain_features(domain, label)
                    logger.debug('domain: %s features %s', domain, features)
                    redis.set(domain, json.dumps(features), ex=14400)
                    redis.publish(ENRICHED_DOMAIN, domain)
                    logger.info('published %s to channel %s', domain, ENRICHED_DOMAIN)
        except ConnectionError:
            logger.exception(message)

if __name__ == "__main__":
    main()
