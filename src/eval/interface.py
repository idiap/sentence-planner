#
# Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
# Written by Andreas Marfurt <andreas.marfurt@idiap.ch>
#

""" Interface for evaluation classes to implement. """

import os
import glob

from abc import ABC, abstractmethod


class Evaluation(ABC):

    def __init__(self, input_dir):
        self.name = None
        self.project = 'nlu'
        self.queue = None
        self.parallel_env = None
        self.conda_env = 'presumm'
        self.input_dir = input_dir

    def get_output_path(self):
        return os.path.join(self.input_dir, self.name + '.csv')

    def get_running_path(self):
        return os.path.join(self.input_dir, self.name + '.running')

    @abstractmethod
    def run(self):
        pass

    def _read_files(self, file_suffix, description):
        """ Reads output files. Only one can be present in the input folder. """
        files = glob.glob(os.path.join(self.input_dir, '*%s' % file_suffix))
        assert len(files) == 1, '%s contains zero or multiple %s files' % (self.input_dir, description)
        path = files[0]
        with open(path, 'r') as f:
            lines = list(map(str.strip, f.readlines()))
        return lines

    def read_candidate_summaries(self):
        return self._read_files('.candidate', 'candidate summary')

    def read_reference_summaries(self):
        return self._read_files('.gold', 'reference summary')

    def read_articles(self):
        return self._read_files('.raw_src', 'article')
