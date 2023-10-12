import tests.utils.e2e.readers
import tests.utils.e2e.preprocessors
import tests.utils.e2e.preprocessors_tf_hub
import tests.utils.e2e.ir_provider
import tests.utils.e2e.infer
import tests.utils.e2e.postprocessors
import tests.utils.e2e.ref_collector
import tests.utils.e2e.model_loader
import tests.utils.e2e.omz_pytorch_to_onnx_converter
from types import SimpleNamespace

from tests.utils.e2e.common.base_provider import BaseStepProvider


class Pipeline:
    def __init__(self, config):
        self._config = config
        self.steps = []
        for name, params in config.items():
            self.steps.append(BaseStepProvider.provide(name, params))
        self.details = SimpleNamespace(xml=None, mo_log=None)

    def run(self):
        try:
            for i, step in enumerate(self.steps):
                step_input = None if i == 0 else self.steps[i - 1].out_data
                if step.__step_name__ == "get_ir" and self.steps[i - 1].__step_name__ == 'preprocess_tf_hub':
                    step_input = self.steps[i - 1].out_data[0]
                step.execute(step_input)

                if step.__step_name__ == 'load_model' or step.__step_name__ == 'tf_hub_load_model':
                    for target_step in self.steps:
                        if target_step.__step_name__ == "get_ir":
                            target_step.executor.prepared_model = step.executor.prepared_model

                # If current step is IR generation it's auxiliary path data and explicitly set xml and bin attributes
                # of 'infer' step executor
                if step.__step_name__ == "get_ir":
                    for target_step in self.steps:
                        if target_step.__step_name__ == "infer":
                            target_step.executor.xml = step.executor.xml
                            target_step.executor.bin = step.executor.bin
                            break
        finally:
            # Handle exception and fill `Pipeline_obj.details` to provide actual information for a caller
            for step in self.steps:
                if step.__step_name__ == "get_ir":
                    self.details.xml = step.executor.xml
                    self.details.mo_log = step.executor.mo_log

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
