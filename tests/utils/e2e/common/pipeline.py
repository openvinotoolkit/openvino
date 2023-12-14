import utils.e2e.readers
import utils.e2e.preprocessors
import utils.e2e.preprocessors_tf_hub
import utils.e2e.ir_provider
import utils.e2e.infer
import utils.e2e.postprocessors
import utils.e2e.ref_collector
import utils.e2e.model_loader
import utils.e2e.omz_pytorch_to_onnx_converter

from utils.e2e.common.base_provider import BaseStepProvider


class Pipeline:
    def __init__(self, config):
        self._config = config
        self.steps = []
        for name, params in config.items():
            self.steps.append(BaseStepProvider.provide(name, params))

    def run(self):
        for i, step in enumerate(self.steps):
            if step.__step_name__ in ["infer", "postprocessor"]:
                step.execute(self.steps[i - 1].out_data)
            else:
                step.execute()

    def fetch_results(self):
        if len(self.steps) == 0:
            # raise ValueError("Impossible to fetch results from an empty pipeline")
            return None
        return self.steps[-1].out_data

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
