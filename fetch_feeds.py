import json
from redis import StrictRedis
from time import sleep
import argparse
import os
from DataSource import OpenPhishDataSource, OPENPHISH_URL, OPENPHISH_STR, NEW_URL_TOPIC
from DataSource import PhishTankDataSource, PHISHTANK_URL, PHISHTANK_STR
from DataSource import AlexaDataSource, ALEXA_URL, ALEXA_STR
from DataSource import OPENDNS_URL, OPENDNS_STR
from common import define_logger

logger = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infinity", action='store_true', help='infinte run')
    parser.add_argument("--sleep", type=int, default=10000, help="sleep seconds, relevant only on infinity mode")
    parser.add_argument("--data-source", type=str, help="data source to fetch")
    parser.add_argument("--redis-host", type=str, default='localhost', help="redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="redis db index")
    parser.add_argument("--limit", type=int, default=10000, help="publish limit for fetched URLs")
    parser.add_argument("--debug-level", type=str, default='INFO', help='logging debug level')
    args = parser.parse_args()
    print(args)
    data_source_arg = args.data_source.lower()
    data_source = None
    if data_source_arg == OPENPHISH_STR.lower():
        data_source = OpenPhishDataSource(OPENPHISH_URL, OPENPHISH_STR, 1, NEW_URL_TOPIC, None)

    elif data_source_arg == PHISHTANK_STR.lower():
        phishtank_apikey = ''
        if 'PHISHTANK_APIKEY' in os.environ:
            phishtank_apikey = os.getenv('PHISHTANK_APIKEY')
        phishtank_url = PHISHTANK_URL.format(apikey=phishtank_apikey)
        data_source = PhishTankDataSource(phishtank_url, PHISHTANK_STR, 1, NEW_URL_TOPIC, None)

    elif data_source_arg == ALEXA_STR.lower():
        data_source = AlexaDataSource(ALEXA_URL, ALEXA_STR, 0, NEW_URL_TOPIC, None)

    elif data_source_arg == OPENDNS_STR.lower():
        data_source = AlexaDataSource(OPENDNS_URL, OPENDNS_STR, 0, NEW_URL_TOPIC, None)
    
    global logger
    logger = define_logger(data_source.get_origin(), args.debug_level)
    logger.info('program args: %s', args)
    logger.info('data_source: %s', data_source.get_origin())
    
    while(True):
        redis = StrictRedis(host=args.redis_host, port=args.redis_port, db=args.redis_db)
        fetch_feed(data_source, args.limit, redis)
        if not args.infinity:
            break
        sleep_duration = args.sleep
        # sleep_duration = int(os.getenv('FETCH_SLEEP_TIME'))
        logger.info('going to sleep now for %d seconds...', sleep_duration)
        sleep(sleep_duration)
        logger.info('woke up!')


def fetch_feed(data_source, publish_limit, redis):
    if data_source:
        origin = data_source.get_origin()
        logger.info('handling %s', origin)
        url_list = data_source.fetch()
        logger.info('len(url_list): %d', len(url_list))
        logger.info('url_list[0]: %s', url_list[0])
        
        latest_url = redis.get(origin)

        if latest_url:
            latest_url = latest_url.decode('utf-8')

        logger.info('latest_url (from redis): %s', latest_url)

        if data_source.get_label() == 0 or latest_url is None or latest_url != url_list[0]:
            last_url_index = len(url_list)
            if data_source.get_label() == 0 or latest_url not in url_list:
                last_url_index = len(url_list) - 1
            else:
                last_url_index = url_list.index(latest_url)
            logger.info('fetch done. got %d new urls', len(url_list))
            logger.info('last_url_index: %d', last_url_index)

            for url in url_list[:min(last_url_index, publish_limit)]:
                message = json.dumps({'url': url, 'label': data_source.get_label()})
                logger.debug('publish message: %s', message)
                redis.publish(data_source.get_topic(), message)

            logger.info('update latest url of %s to: %s', origin, url_list[0])
            redis.set(origin, url_list[0], ex=86400)

if __name__ == "__main__":
    main()
