import itertools
import os
from collections import OrderedDict

from e2e_oss.common_utils.tf_helper import TFVersionHelper
from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.collect_reference_templates import get_refs_tf
from e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_oss._utils.path_utils import prepend_with_env_path


class DynamicSequenceLengthBase(CommonConfig):
    """
    Test class for synthetic models with dynamic sequence lengths.
    """
    align_results = None

    def __init__(self, batch, device, precision, api_2, **kwargs):
        if "BATCH={}".format(batch) not in self.model:
            self.__do_not_run__ = True

        input_file = "test_data/inputs/tf/DynamicSequenceLength_batch_{}.npz".format(batch)
        model_path = prepend_with_env_path("tf_internal_models", TFVersionHelper().tf_models_version, "synthetic_lstm", self.model)
        output_nodes = 'Reshape'
        infer_api = 'ie_sync_api_2' if api_2 else 'ie_sync'

        self.ref_pipeline = OrderedDict([
            read_npz_input(path=input_file),
            get_refs_tf(
                model=model_path, additional_outputs=['Reshape'])
        ])
        self.ie_pipeline = OrderedDict([
            # 1. Read Input data
            read_npz_input(path=input_file),
            # 3. Generate ir
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path,
                                 precision=precision,
                                 input='data,Placeholder',
                                 input_shape='[{batch},20,16],[{batch}]'.format(batch=batch),
                                 output=output_nodes),
            # 4. Run inference with IE
            ("infer", {infer_api: {"device": device, "cpu_extension": "cpu_extension"}}),
        ])
        self.comparators = eltwise_comparators(precision=precision, device=device,
                                               target_layers=output_nodes.split(','))


def class_factory(cls_name, cls_kwargs, BaseClass=DynamicSequenceLengthBase):
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


for item in itertools.product([1, 2], ['lstm', 'gru', 'rnn'], ['dynamic'],
                              ['placeholder'], [False]):
    (batch, cell_name, rnn_name, seq_len_name, time_major) = item
    model = 'CELL={}_RNN={}_SEQLEN={}_TIME-MAJOR={}_BATCH={}'.format(cell_name, rnn_name,
                                                                     seq_len_name,
                                                                     str(time_major),
                                                                     str(batch))
    class_name = "DynamicSequenceLength_" + model
    locals()[class_name] = class_factory(cls_name=class_name,
                                         cls_kwargs={"__is_test_config__": True,
                                                     "model": os.path.join(model, "frozen.pb")})
