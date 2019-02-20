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

    def update_ranges(self, ranges):
        assert self.dims == ranges.shape[0]
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
        ret += '>'
        return ret



class holey:


    supported_input_data_types = ['numpy', 'doc2vec']
    supported_rect_expansion_strategies = ['randmax']

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


    def __init__(self, input_data, input_data_type='doc2vec'):

        if input_data_type not in self.supported_input_data_types:
            error(f'Input data type not supported: {input_data_type}')

        self.load_data(input_data, input_data_type)


    def load_data(self, input_data, input_data_type='doc2vec'):

        if input_data_type not in self.supported_input_data_types:
            error(f'Input data type not supported: {input_data_type}')

        # data types
        if input_data_type == 'doc2vec':
            self.load_data_doc2vec(input_data)
        elif input_data_type == 'numpy':
            self.load_data_numpy(input_data)
        else:
            error(f'FATAL: No loading function implemented for data type: {input_data_type}')


    def load_data_numpy(self, input_data):

        assert len(input_data.shape) == 2
        self.samples, self.dims = input_data.shape
        self.data = input_data.copy()
        log(f'Loaded numpy array data: {self.samples} vectors of length {self.dims}.')


    def load_data_doc2vec(self, input_data):

        from gensim.models import Doc2Vec

        try:
            model = Doc2Vec.load(input_data)
        except:
            error(f'Failed to load doc2vec model from file: {input_data}')

        vec_count = model.docvecs.count
        vec_len = model.docvecs.vector_size
        self.samples = vec_count
        self.dims = vec_len

        self.data = model.docvecs.vectors_docs
        log(f'Loaded doc2vec data: {self.samples} vectors of length {self.dims}.')


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


    def check_if_point_in_rectangle(self, rect, point):

        for dim in range(rect.dims):
            if point[dim] < rect.ranges[dim][0] or point[dim] > rect.ranges[dim][1]:
                return False
        # contained in all dimensions
        return True


    def check_if_point_in_ranges(self, ranges, point):

        for dim in range(ranges.shape[0]):
            if point[dim] < ranges[dim][0] or point[dim] > ranges[dim][1]:
                return False
        # contained in all dimensions
        return True


    def check_if_no_data_in_ranges(self, ranges):

        for point in self.data:
            if self.check_if_point_in_ranges(ranges, point):
                # found a point inside the ranges
                return False
        # no points inside the ranges
        return True


    def get_surrounding_edges(self, value, dim):
        for i, edge in enumerate(self.projs[dim]):
            if edge > value:
                assert value > self.projs[dim][i-1]
                return self.projs[dim][i-1], edge



    def expand_rectangle(self, rect, dimension, direction, amount='maximal'):
        '''Expand a rectangle along one dimension in one direction by some amount.
        Stop expanding at data points (rectangle would be non-empty).
        If amount == 'maximal', expand until a data point is encountered.
        '''

        assert dimension < rect.dims
        up = None
        if direction == '+':
            up = True
        elif direction == '-':
            up = False
        else:
            error(f'Rectangle expansion direction must be either + or - .')
        
        if amount != 'maximal':
            error('Specific amount expansion NYI')
            # if amount <= 0 or amount > 1:
            #     error(f'Rectangle expansion amount must be in (0, 1]  - given: {amount}')

        ranges = rect.ranges
        done = False
        if up:
            edge = ranges[dimension][1]

            # find next step on orthogonal projection
            for val in self.projs[dimension]:
                # skip lower values
                if val <= edge:
                    continue

                # val is outside rectangle, expand
                ranges[dimension][1] = val

                # check if expansion is ok
                for point in self.data:
                    
                    if self.check_if_point_in_ranges(ranges, point):
                        # expanded too far, revert
                        ranges[dimension][1] = edge
                        done = True
                        break

                if done:
                    break

        else:
            edge = ranges[dimension][0]

            # find next step on orthogonal projection
            for val in self.projs[dimension][::-1]:
                # skip higher values
                if val >= edge:
                    continue

                # val is outside rectangle, expand
                ranges[dimension][0] = val

                # check if expansion is ok
                for point in self.data:
                    
                    if self.check_if_point_in_ranges(ranges, point):
                        # expanded too far, revert
                        ranges[dimension][0] = edge
                        done = True
                        break

                if done:
                    break
        

        rect.update_ranges(ranges)
        # log(f'Expanded to {rect.ranges[dimension]} in direction {direction} on dimension {dimension}.')
        return rect


    def maximize_rectangle_randmax(self, rect):
        '''Expand dimensions in random order, each maximally'''

        # log(f'>>> Starting expansion with rect: {rect}')

        # i+dims are dummies for negative directions
        dimorder = np.random.permutation(self.dims*2)
        for dim in dimorder:
            if dim < self.dims:
                rect = self.expand_rectangle(rect, dim, '+')
            else:
                rect = self.expand_rectangle(rect, dim-self.dims, '-')

        # log(f'Ended expansion with rect: {rect}\n#############')

        return rect



    def maximize_rectangle(self, rect, expansion_strategy):
        '''Expand rectangle to be maximal

        Supported expansion strategies:
        - randmax:  random order of dims, each maximally
        '''

        if expansion_strategy not in self.supported_rect_expansion_strategies:
            error(f'Unknown rectangle expansion strategy: {expansion_strategy}')

        if expansion_strategy == 'randmax':
            return self.maximize_rectangle_randmax(rect)       

        else:
            error(f'Rectangle expansion strategy NYI: {expansion_strategy}')


    def make_rectangle_monte_carlo(self, expansion_strategy='randmax'):

        if expansion_strategy not in self.supported_rect_expansion_strategies:
            error(f'Unknown rectangle expansion strategy: {expansion_strategy}')

        # throw a random dart at the board    >--~-  ((o))
        dart = np.random.rand(self.dims)

        # find initial empty rectangle
        ranges = np.zeros((self.dims, 2))
        for dim in range(self.dims):
            ranges[dim] = self.get_surrounding_edges(dart[dim], dim)
        rect = holeyRect(ranges)

        # expand
        rect = self.maximize_rectangle(rect, expansion_strategy)

        assert self.check_if_point_in_rectangle(rect, dart)

        return rect
            


    def generate_rectangles(self, num_tries=10000, num_rectangles=None, size_threshold='default_1_over_n', random_seed=42):

        assert self.data_isnormalized, 'Normalize data before running'
        assert self.projs is not None, 'Get orthogonal projections before running'

        assert num_rectangles is None, 'Target number of rectangles NYI'


        # init
        if size_threshold == 'default_1_over_n':
            size_threshold = 1/self.dims
        elif size_threshold < 0 or size_threshold >= 1:
            error(f'Rectangle size threshold must be in [0, 1)  - given: {size_threshold}')

        max_volume_found = 0
        current_tries = 0
        rectangle_list = []
        np.random.seed(random_seed)


        # main loop
        current_rectangle = None
        while current_tries < num_tries:


            current_rectangle = self.make_rectangle_monte_carlo()

            # size filter
            if current_rectangle.volume < size_threshold:
                current_tries += 1
                continue

            # new max, add to result and reset tries
            if current_rectangle.volume > max_volume_found:
                max_volume_found = current_rectangle.volume
                rectangle_list.append(current_rectangle)
                current_tries = 0

            else:
                current_tries += 1

        return rectangle_list




################

if __name__ == '__main__':

    log('Started holey test run ...')

    log('Loading model ...')
    # load model
    # model = '/home/ya86gul/scripts/holey/nanotext/nanotext/data/embedding.genomes.model'
    # holey = holey(model)

    # # subset model
    # holey.data = holey.data[:100]

    ho = holey(np.random.randn(100, 10), 'numpy')
    # holey = holey(np.array([3,5,7,9,5,4,7,5,3,4,9,6,4,2,4], dtype=float).reshape((5,3)), 'numpy')

    print(ho.data[:5])
    log('Normalizing data ...')

    ho.normalize_data()
    print(ho.data[:5])

    # log('Unnormalizing data ...')
    # holey.unnormalize_data()
    # print(holey.data[:5])

    ho.get_orthogonal_projections()
    print(ho.projs_lens)


    rects = ho.generate_rectangles(num_tries=100)

    # check them again
    for rect in rects:
        print(rect)
        empty = ho.check_if_no_data_in_ranges(rect.ranges)
        print(f'Rectangle is empty: {empty}')

    log('Done.')
