import pickle
import logging

import config


class Preprocess:
    def __init__(self):
        self.logger = logging.getLogger('preprocess')
    
    def save_data(self, save_data, save_name):
        self.logger.info('saving data')
        pickle.dump(save_data, open(config.PROCESSED_DATASET_PATH + save_name, 'wb'))
