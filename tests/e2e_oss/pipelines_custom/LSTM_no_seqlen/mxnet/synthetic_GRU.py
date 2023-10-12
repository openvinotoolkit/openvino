from collections import OrderedDict
from pathlib import Path

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.collect_reference_templates import get_refs_mxnet, read_refs_pipeline
from e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from tests.e2e_oss._utils.path_utils import prepend_with_env_path, ref_from_model
from utils.pytest_utils import mark, timeout


class MXNET_Synthetic_GRU_Base(CommonConfig):

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


class MXNET_Synthetic_GRU_Simple(MXNET_Synthetic_GRU_Base):
    __is_test_config__ = True
    ref_collection = True
    model_name = "mxnet_synthetic_gru_simple"
    model = 'synthetic_lstm_tests/GRU/simple_model/net.params-0000.params'
    symbol = 'synthetic_lstm_tests/GRU/simple_model/net.params-symbol.json'


class MXNET_Synthetic_GRU_Bidirectional(MXNET_Synthetic_GRU_Base):
    __is_test_config__ = True
    ref_collection = True
    model_name = "mxnet_synthetic_gru_biderctional"
    model = 'synthetic_lstm_tests/GRU/bidirectional/net.params-0000.params'
    symbol = 'synthetic_lstm_tests/GRU/bidirectional/net.params-symbol.json'

    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__(batch, device, precision, api_2, **kwargs)

        self.__pytest_marks__ += tuple([mark(timeout(600, "Not enough time for infer"))])


class MXNET_Synthetic_GRU_Multilayer(MXNET_Synthetic_GRU_Base):
    __is_test_config__ = True
    ref_collection = True
    model_name = "mxnet_synthetic_gru_multilayer"
    model = 'synthetic_lstm_tests/GRU/multilayer/net.params-0000.params'
    symbol = 'synthetic_lstm_tests/GRU/multilayer/net.params-symbol.json'
