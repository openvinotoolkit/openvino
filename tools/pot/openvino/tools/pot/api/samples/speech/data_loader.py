# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import struct
import numpy as np

from openvino.tools.pot import DataLoader


class ArkDataLoader(DataLoader):

    def __init__(self, config):
        super().__init__(config)
        self._data_source = config.get('data_source')
        self._shape = None
        self._ark_frames_num = []
        self._feed_dict = None
        self._feed_dict, self._shapes_dict = self._create_feed_dicts(config)
        first_input_name = list(self._feed_dict.keys())[0]
        for ark_name in self._feed_dict[first_input_name]:
            num_frames = self._get_num_frames_from_ark(ark_name)
            self._ark_frames_num.append(num_frames)

    def _create_feed_dicts(self, config):
        feed_dict = {name: [] for name in config['input_names']}
        shapes_dict = {}
        ark_filenames = self._collect_ark_filenames(self._data_source)
        for ark_filename in ark_filenames:
            for input_id, input_name in enumerate(feed_dict.keys()):
                if config['input_shapes']:
                    shapes_dict[input_name] = config['input_shapes'][input_id]
                if 'input_files' in config:
                    input_filename = config['input_files'][input_id] + '.ark'
                    if ark_filename == input_filename:
                        feed_dict[input_name].append(os.path.join(self._data_source, input_filename))
                    else:
                        continue
                else:
                    feed_dict[input_name].append(os.path.join(os.path.dirname(self._data_source), ark_filename))

        return feed_dict, shapes_dict

    def __getitem__(self, frame_id):
        ark_file_id = 0
        first_frame_num = 0
        output_feed_dict = {}
        for input_name in self._feed_dict:
            for file_id, _ in enumerate(self._feed_dict[input_name]):
                if frame_id < first_frame_num + self._ark_frames_num[file_id]:
                    break
                first_frame_num += self._ark_frames_num
                ark_file_id = file_id
            index_id = frame_id - first_frame_num
            output_feed_dict[input_name] = self._read_frames_from_ark(self._feed_dict[input_name][ark_file_id],
                                                                      index_id)
            if input_name in self._shapes_dict:
                output_feed_dict[input_name] = np.reshape(output_feed_dict[input_name], self._shapes_dict[input_name])

        return output_feed_dict

    def __len__(self):
        return sum(self._ark_frames_num)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = tuple(shape)

    @staticmethod
    def _collect_ark_filenames(data_source):
        ark_filenames = []
        if os.path.isdir(data_source):
            for filename in os.listdir(data_source):
                if filename.endswith('.ark'):
                    ark_filenames.append(filename)
        elif os.path.isfile(data_source):
            if data_source.endswith('.ark'):
                ark_filenames.append(os.path.basename(data_source))
        if not ark_filenames:
            raise FileNotFoundError('Ark files not found!')
        return ark_filenames

    def _get_num_frames_from_ark(self, ark_filename):
        frames_num = 0
        with open(ark_filename, 'rb') as fd:
            while True:
                try:
                    key = self.read_token(fd)
                    if not key:
                        break
                    fd.peek(4)
                    ark_type = self.read_token(fd)
                    float_size = 4 if ark_type[2] == 'F' else 8
                    num_rows = self.read_int32(fd)
                    num_cols = self.read_int32(fd)
                    fd.read(float_size * num_cols * num_rows)
                    frames_num += num_rows
                except EOFError:
                    break
        return frames_num

    def _read_frames_from_ark(self, ark_filename, frame_index):
        with open(ark_filename, 'rb') as fd:
            ut = {}
            while True:
                try:
                    key = self.read_token(fd)
                    if not key:
                        break
                    fd.peek(4)
                    ark_type = self.read_token(fd)
                    float_size = 4 if ark_type[2] == 'F' else 8
                    float_type = np.float32 if ark_type[2] == 'F' else float
                    num_rows = self.read_int32(fd)
                    num_cols = self.read_int32(fd)
                    mat_data = fd.read(float_size * num_cols * num_rows)
                    mat = np.frombuffer(mat_data, dtype=float_type)
                    ut[key] = mat.reshape(num_rows, num_cols)
                except EOFError:
                    break
        return np.concatenate(list(ut.values()), 0)[frame_index]

    @staticmethod
    def read_token(fd):
        key = ''
        while True:
            c = bytes.decode(fd.read(1))
            if c in [' ', '']:
                break
            key += c
        return None if key == '' else key.strip()

    @staticmethod
    def read_int32(fd):
        int_size = bytes.decode(fd.read(1))
        if int_size != '\04':
            raise AssertionError('Expect \'\\04\', but gets {}'.format(int_size))
        int_str = fd.read(4)
        int_val = struct.unpack('i', int_str)
        return int_val[0]
