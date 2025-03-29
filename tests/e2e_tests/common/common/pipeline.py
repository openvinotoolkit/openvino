# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import e2e_tests.common.readers
import e2e_tests.common.preprocessors
import e2e_tests.common.preprocessors_tf_hub
import e2e_tests.common.ir_provider
import e2e_tests.common.infer
import e2e_tests.common.postprocessors
import e2e_tests.common.ref_collector
import e2e_tests.common.model_loader
from e2e_tests.common.common.base_provider import BaseStepProvider
from types import SimpleNamespace


class PassThroughData(dict):
    """
    Syntactic sugar around standard dictionary class.
    Encapsulates error handling while working with passthrough_data in StepProvider classes
    """
    def strict_get(self, key, step):
        assert key in self, \
            "Step `{}` requires `{}` key to be defined by previous steps".format(step.__step_name__,  key)
        return self.get(key)


class Pipeline:

    def __init__(self, config, passthrough_data=None):
        self._config = config
        self.steps = []
        for name, params in config.items():
            self.steps.append(BaseStepProvider.provide(name, params))
        self.details = SimpleNamespace(xml=None, mo_log=None)
        # passthrough_data delivers necessary data from / to steps including first step
        # it doesn't have any restriction on steps being consecutive to pass the data
        # steps are allowed to read and write to passthrough_data
        self.passthrough_data = PassThroughData() if passthrough_data is None else PassThroughData(passthrough_data)

    def run(self):
        try:
            for i, step in enumerate(self.steps):
                self.passthrough_data = step.execute(self.passthrough_data)
        finally:
            # Handle exception and fill `Pipeline_obj.details` to provide actual information for a caller
            self.details.xml = self.passthrough_data.get('xml', None)
            self.details.mo_log = self.passthrough_data.get('mo_log', None)

    def fetch_results(self):
        if len(self.steps) == 0:
            # raise ValueError("Impossible to fetch results from an empty pipeline")
            return None
        return self.passthrough_data.get('output', None)

    def fetch_test_info(self):
        if len(self.steps) == 0:
            return None
        test_info = {}
        for step in self.steps:
            info_from_step = getattr(step, "test_info", {})
            assert len(set(test_info.keys()).intersection(info_from_step.keys())) == 0,\
                'Some keys have been overwritten: {}'.format(set(test_info.keys()).intersection(info_from_step.keys()))
            test_info.update(info_from_step)
        return test_info
