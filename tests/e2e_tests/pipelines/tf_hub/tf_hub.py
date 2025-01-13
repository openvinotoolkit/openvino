# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from e2e_tests.test_utils.test_utils import class_factory
from e2e_tests.pipelines.production.tf_hub_case_class import TFHUB_eltwise_Base


def get_models_list(file_name: str):
    models = []
    with open(file_name) as f:
        for model_info in f:
            # skip comment in model scope file
            if model_info.startswith('#'):
                continue
            mark = None
            reason = None
            assert len(model_info.split(',')) == 2 or len(model_info.split(',')) == 4, \
                "Incorrect model info `{}`. It must contain either 2 or 4 fields.".format(model_info)
            if len(model_info.split(',')) == 2:
                model_name, model_link = model_info.split(',')
            elif len(model_info.split(',')) == 4:
                model_name, model_link, mark, reason = model_info.split(',')
                assert mark == "skip", "Incorrect failure mark for model info {}".format(model_info)
            models.append((model_name, model_link.strip(), mark, reason))

    return models


model_files = ['precommit']
models = []
for file in model_files:
    models += get_models_list(os.path.join(os.path.dirname(__file__), f"{file}.yml"))

base_class = TFHUB_eltwise_Base

for model in models:
    class_name = model[0]
    model_link = model[1]
    if sys.platform == 'win32':
        model_link = model_link.split('?')[0]
    locals()[class_name] = class_factory(cls_name=class_name,
                                         cls_kwargs={'__is_test_config__': True,
                                                     'model_name': class_name,
                                                     'model_link': model_link},
                                         BaseClass=base_class)


