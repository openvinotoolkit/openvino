# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict


class TestInfo:
    extra_info = None
    e2e_models_info = None

    def fill_mo_args(self, **mo_params):
        if self.extra_info is None:
            self.extra_info = defaultdict(list)
        if self.e2e_models_info is None:
            self.e2e_models_info = defaultdict(list)

        if self.extra_info.get('model_optimizer_args', None) is None:
            self.extra_info.update({'model_optimizer_args': {}})

        for mo_key in mo_params.keys():
            if mo_key == 'additional_args':
                for add_arg, add_val in mo_params['additional_args'].items():
                    self.fill_mo_args(**{add_arg: add_val})
            else:
                self.extra_info['model_optimizer_args'].update({mo_key: str(mo_params[mo_key])})

    def fill_extra_info(self, op_type, name, op, shape, dtype):
        if self.extra_info is None:
            self.extra_info = defaultdict(list)
        if self.e2e_models_info is None:
            self.e2e_models_info = defaultdict(list)

        if op_type == 'Const':
            self.extra_info['Constants'].append({'name': name,
                                                 'op': op,
                                                 'shape': shape,
                                                 'dtype': dtype})

        if op_type == 'FakeConst':
            self.extra_info['FakeConstants'].append({'name': name,
                                                     'op': op,
                                                     'shape': shape,
                                                     'dtype': dtype})

        if op_type == 'Variable':
            self.extra_info['Variables'].append({'name': name,
                                                 'op': op,
                                                 'shape': shape,
                                                 'dtype': dtype})

        if op_type == 'Input':
            self.extra_info['Inputs'].append({'name': name,
                                              'op': op,
                                              'shape': shape,
                                              'dtype': dtype})
        if op_type == 'Intermediate':
            self.extra_info['Intermediates'].append({'name': name,
                                                     'op': op,
                                                     'shape': shape,
                                                     'dtype': dtype})

        if op_type == 'Output':
            self.extra_info['Outputs'].append({'name': name,
                                               'op': op,
                                               'shape': shape,
                                               'dtype': dtype})
