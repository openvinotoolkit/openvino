import itertools
import os
from collections import OrderedDict

from tests.e2e_oss._utils.path_utils import prepend_with_env_path
from tests.e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from tests.e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from tests.e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from tests.e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation


class ONNX_Synthetic_LSTM_Base(CommonConfig):
    align_results = None

    def __init__(self, batch, device, precision, api_2, **kwargs):
        if "batch={}".format(batch) not in self.model:
            self.__do_not_run__ = True

        model_path = prepend_with_env_path("onnx_internal_models", "synthetic_lstm_onnx", self.model)
        infer_api = 'ie_sync_api_2' if api_2 else 'ie_sync'

        self.ref_pipeline = OrderedDict([
            ("get_refs", {"precollected": {"path": os.path.join(os.path.dirname(model_path), "reference.npz")}})
        ])
        self.ie_pipeline = OrderedDict([
            # 1. Read Input data
            read_npz_input(os.path.join(os.path.dirname(model_path), "input.npz")),
            # 2. Generate ir
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path,
                                 precision=precision),
            # 3. Run inference with IE
            ("infer", {infer_api: {"device": device, "cpu_extension": "cpu_extension"}})
        ])
        self.comparators = eltwise_comparators(precision=precision, target_layers=["output"], device=device)


def class_factory(cls_name, cls_kwargs, BaseClass=ONNX_Synthetic_LSTM_Base):
    """
    Function that generates of custom classes
    :param cls_name: name of the future class
    :param cls_kwargs: attributes required for the class (e.g. __is_test_config__)
    :param BaseClass: basic class where implemented behaviour of the test
    :return:
    """

    # Generates new class with "cls_name" type inherited from "object" and
    # with specified "__init__" and other class attributes
    newclass = type(cls_name, (BaseClass,), {**cls_kwargs})
    return newclass


for item in itertools.product([1, 2], ['LSTM', 'GRU', 'RNN'], [False, True]):
    batch, cell_type, bidirectional = item
    model = 'cell={}_bidirectional={}_batch={}'.format(cell_type, str(bidirectional), batch)
    class_name = "ONNX_Synthetic_LSTM_" + model
    locals()[class_name] = class_factory(cls_name=class_name,
                                         cls_kwargs={"__is_test_config__": True,
                                                     "model": os.path.join(model, "model.onnx")})
