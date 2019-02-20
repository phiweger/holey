#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
holey
find large empty rectangles in big data

SK
'''

import sys
import numpy as np


def error(string, error_type=1):
    sys.stderr.write(f'ERROR: {string}\nAborting.\n')
    sys.exit(error_type)


def log(string, newline_before=False):
    if newline_before:
        sys.stderr.write('\n')
    sys.stderr.write(f'LOG: {string}\n')


#####


class holeyRect:

    dims = None
    ranges = None
    volume = None

    def __init__(self, ranges):
        self.dims = ranges.shape[0]
        self.ranges = ranges
        self.volume = self.get_volume()

    def get_volume(self):
        assert self.ranges is not None
        return np.prod(np.ptp(self.ranges, 1))

    def __repr__(self):
        ret = f'<holeyRect of {self.dims} dimensions:\n\tVolume: {self.volume}\n'
        for i, dim in enumerate(range(self.dims)):
            if i >= 20:
                ret += '\t...\n'
                break
            ret += f'\t( {self.ranges[dim][0]}, \t{self.ranges[dim][1]} )\n'
        ret += '>\n'
        return ret



class holey:


    supported_input_data_types = ['doc2vec']

    data = None
    samples = None
    dims = None

    data_isnormalized = False
    # save normalization params for each column
    normalized_shift = None
    normalized_scale = None


    # sorted orthogonal projections
    projs = []
    projs_lens = []


    def __init__(self, input_file):

        self.load_data(input_file)


    def load_data(self, input_file, input_data_type='doc2vec'):

        if input_data_type not in self.supported_input_data_types:
            error(f'Input data type not supported: {input_data_type}')

        # data types
        if input_data_type == 'doc2vec':
            self.load_data_doc2vec(input_file)

        else:
            error(f'FATAL: No loading function implemented for data type: {input_data_type}')


    def load_data_doc2vec(self, input_file):

        from gensim.models import Doc2Vec

        try:
            model = Doc2Vec.load(input_file)
        except:
            error(f'Failed to load doc2vec model from file: {input_file}')

        vec_count = model.docvecs.count
        vec_len = model.docvecs.vector_size
        self.samples = vec_count
        self.dims = vec_len

        self.data = model.docvecs.vectors_docs
        log(f'Loaded doc2vec data: {vec_count} vectors of length {vec_len}.')


    def normalize_data(self):

        if self.data_isnormalized:
            error(f'Attempt to normalize data, but it is already normalized.')
        assert self.normalized_shift is None
        assert self.normalized_scale is None

        self.normalized_shift = np.amin(self.data, axis=0)
        self.normalized_scale = np.ptp(self.data, axis=0)
        # ptp: 'peak to peak' range, abs(max-min)

        for col in range(self.dims):
            self.data[:, col] = (self.data[:, col] - self.normalized_shift[col]) / self.normalized_scale[col]

        self.data_isnormalized = True


    def unnormalize_data(self):

        if not self.data_isnormalized:
            error(f'Attempt to unnormalize data, but it is not normalized.')
        assert self.normalized_shift is not None
        assert self.normalized_scale is not None

        for col in range(self.dims):
            self.data[:, col] = (self.data[:, col] * self.normalized_scale[col]) + self.normalized_shift[col]

        self.data_isnormalized = False


    def get_orthogonal_projections(self):

        assert self.data_isnormalized

        for col in range(self.dims):
            ortho = np.unique(self.data[:, col])
            self.projs.append(ortho)
            self.projs_lens.append(ortho.size)


#############


    def get_surrounding_edges(self, value, dim):
        for i, edge in enumerate(self.projs[dim]):
            if edge > value:
                assert value > self.projs[dim][i-1]
                return self.projs[dim][i-1], edge


    def make_rectangle_monte_carlo(self):

        # throw a random dart at the board    >--~-  ((o))
        dart = np.random.rand(self.dims)

        # find initial empty rectangle
        ranges = np.zeros((self.dims, 2))
        for dim in range(self.dims):
            ranges[dim] = self.get_surrounding_edges(dart[dim], dim)
        rect = holeyRect(ranges)
        
        # expand



        return rect
            


    def generate_rectangles(self, num_tries=10000, num_rectangles=None, size_threshold='default_1_over_n', random_seed=42):

        assert self.data_isnormalized, 'Normalize data before running'
        assert self.projs is not None, 'Get orthogonal projections before running'

        assert num_rectangles is None, 'Target number of rectangles NYI'


        # init
        if size_threshold == 'default_1_over_n':
            size_threshold = 1/self.dims

        max_volume_found = 0
        current_tries = 0
        rectangle_list = []
        np.random.seed(random_seed)


        # main loop
        current_rectangle = None
        while current_tries < num_tries:

            current_rectangle = self.make_rectangle_monte_carlo()

            if current_rectangle.volume > max_volume_found:
                # new hit, append and reset
                max_volume_found = current_rectangle.volume
                rectangle_list.append(current_rectangle)
                current_tries = 0

            else:
                # count failed try if below size threshold
                if current_rectangle.volume < size_threshold:
                    current_tries += 1


        return rectangle_list




################

if __name__ == '__main__':

    log('Started holey test run ...')

    # load model
    model = '/home/ya86gul/scripts/holey/nanotext/nanotext/data/embedding.genomes.model'
    log('Loading model ...')
    holey = holey(model)

    # subset model
    holey.data = holey.data[:100]

    print(holey.data[:2])
    log('Normalizing data ...')

    holey.normalize_data()
    print(holey.data[:2])

    # log('Unnormalizing data ...')
    # holey.unnormalize_data()
    # print(holey.data[:2])

    holey.get_orthogonal_projections()
    print(holey.projs_lens)


    rects = holey.generate_rectangles(10)
    print(rects)

    log('Done.')
