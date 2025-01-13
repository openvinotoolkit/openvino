# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from e2e_tests.test_utils.modify_configs import ie_sbs_reshape_config, ie_reshape_config, mo_reshape_config
from e2e_tests.common.common.pipeline import Pipeline


def ie_pipeline_runner(instance_ie_pipeline, shapes, test_name):
    log.info('Executing IE reshape pipeline for {}'.format(test_name))
    ie_reshape_pipeline = ie_reshape_config(instance_ie_pipeline, shapes, test_name)
    ie_reshape_pipeline = Pipeline(ie_reshape_pipeline)
    ie_reshape_pipeline.run()

    return ie_reshape_pipeline


def mo_pipeline_runner(instance_ie_pipeline, shapes, test_name):
    log.info('Executing MO reshape pipeline for {}'.format(test_name))
    mo_reshape_pipeline = mo_reshape_config(instance_ie_pipeline, shapes, test_name)
    mo_reshape_pipeline = Pipeline(mo_reshape_pipeline)
    mo_reshape_pipeline.run()

    return mo_reshape_pipeline
