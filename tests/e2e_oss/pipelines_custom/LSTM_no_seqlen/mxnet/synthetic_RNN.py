from collections import OrderedDict
from pathlib import Path

from tests.e2e_oss._utils.path_utils import prepend_with_env_path, ref_from_model
from tests.e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from tests.e2e_oss.pipelines.pipeline_templates.collect_reference_templates import get_refs_mxnet, read_refs_pipeline
from tests.e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from tests.e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from tests.e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation


class MXNET_Synthetic_RNN_Base(CommonConfig):
    def __init__(self, batch, device, precision, api_2, **kwargs):
        infer_api = 'ie_sync_api_2' if api_2 else 'ie_sync'
        input_file = str(Path("test_data/inputs/mxnet/synthetic_lstm_input_no_batch.npz"))

        model_env_key = "mxnet_internal_models"
        params = prepend_with_env_path(model_env_key, self.model)
        symbol = prepend_with_env_path(model_env_key, self.symbol)

        self.ref_collection = {'pipeline': OrderedDict([
            read_npz_input(path=input_file),
            ("preprocess", {"align_with_batch": {"batch": 1}}),
            get_refs_mxnet(symbol=symbol, params=params),
            ("postprocess", {"align_with_batch": {"batch": batch}})
        ]),
            'store_path': ref_from_model(self.model_name, framework='mxnet'),
            'store_path_for_ref_save': ref_from_model(self.model_name, framework='mxnet', check_empty_ref_path=False)
        }

        self.ref_pipeline = read_refs_pipeline(ref_file=ref_from_model(model_name=self.model_name, framework='mxnet'),
                                               batch=batch)

        self.ie_pipeline = OrderedDict([
            # 1. Read Input data
            read_npz_input(path=input_file),
            ("preprocess", {"align_with_batch": {"batch": batch}}),
            # 3. Generate ir
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(model_env_key, self.model),
                                 precision=precision,
                                 input_shape=(batch, 10, 16),
                                 legacy_mxnet_model=True),
            # 4. Run inference with IE
            ("infer", {infer_api: {"device": device}}),
        ])
        self.comparators = eltwise_comparators(precision=precision, device=device,
                                               target_layers=['hybridsequential0_relu0_fwd'])


class MXNET_Synthetic_RNN_Simple(MXNET_Synthetic_RNN_Base):
    __is_test_config__ = True
    ref_collection = True
    model_name = "synthetic_rnn_simple"
    model = 'synthetic_lstm_tests/RNN/simple_model/net.params-0000.params'
    symbol = 'synthetic_lstm_tests/RNN/simple_model/net.params-symbol.json'


class MXNET_Synthetic_RNN_Bidirectional(MXNET_Synthetic_RNN_Base):
    __is_test_config__ = True
    ref_collection = True
    model_name = "synthetic_rnn_bidirectional"
    model = 'synthetic_lstm_tests/RNN/bidirectional/net.params-0000.params'
    symbol = 'synthetic_lstm_tests/RNN/bidirectional/net.params-symbol.json'


class MXNET_Synthetic_RNN_Multilayer(MXNET_Synthetic_RNN_Base):
    __is_test_config__ = True
    ref_collection = True
    model_name = "synthetic_rnn_multilayer"
    model = 'synthetic_lstm_tests/RNN/multilayer/net.params-0000.params'
    symbol = 'synthetic_lstm_tests/RNN/multilayer/net.params-symbol.json'


class MXNET_Synthetic_RNN_RELU_Simple(MXNET_Synthetic_RNN_Base):
    __is_test_config__ = True
    ref_collection = True
    model_name = "synthetic_rnn_relu_simple"
    model = 'synthetic_lstm_tests/RNN_relu/simple_model/net.params-0000.params'
    symbol = 'synthetic_lstm_tests/RNN_relu/simple_model/net.params-symbol.json'


class MXNET_Synthetic_RNN_RELU_Bidirectional(MXNET_Synthetic_RNN_Base):
    __is_test_config__ = True
    ref_collection = True
    model_name = "synthetic_rnn_relu_bidrectional"
    model = 'synthetic_lstm_tests/RNN_relu/bidirectional/net.params-0000.params'
    symbol = 'synthetic_lstm_tests/RNN_relu/bidirectional/net.params-symbol.json'


class MXNET_Synthetic_RNN_RELU_Multilayer(MXNET_Synthetic_RNN_Base):
    __is_test_config__ = True
    ref_collection = True
    model_name = "synthetic_rnn_relu_multilayer"
    model = 'synthetic_lstm_tests/RNN_relu/multilayer/net.params-0000.params'
    symbol = 'synthetic_lstm_tests/RNN_relu/multilayer/net.params-symbol.json'
