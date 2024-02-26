import numpy as np
import csv

import os
CURRENT_PATH= os.path.abspath(os.path.join(os.getcwd()))


class RecordBuffer():
    def __init__(self,max_record_buffer_size,design_iterations,designs,exptid):
        self.max_record_buffer_size = max_record_buffer_size
        self.design_iterations = design_iterations
        self.designs = designs
        self.exptid = exptid
        self.mean_rewards_list = np.zeros((self.max_record_buffer_size,1))
        self.clear()

    def clear(self):
        self._top = 0
        self._size = 0

    def _advance(self):
        self._top = (self._top + 1) % self.max_record_buffer_size
        if self._size < self.max_record_buffer_size:
            self._size += 1

    def add_sample(self,mean_rewards=None):
        self.mean_rewards_list[self._top] = mean_rewards
        self._advance()

    def record_to_csv(self):
        data_dir = CURRENT_PATH + "/" + self.exptid
        try:
            os.makedirs(data_dir)
        except:
            pass
        with open(data_dir + '/mean_rewards_design_{}.csv'.format(self.design_iterations),'w',newline='') as file_obj:
            writer = csv.writer(file_obj)
            # record the designs
            writer.writerow(self.designs)
            # record the rewards
            for i in range(1):
                writer.writerow(self.mean_rewards_list[:,i])




