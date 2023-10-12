from . import precollected
from .dummy_ref_collector import use_dummy
from .provider import StepProvider
from .tf_hub_ref_provider import TFHubStepProvider

try:
    from .score_caffe import ScoreCaffe
except ImportError as error:
    ScoreCaffe = use_dummy('score_caffe', error_message=str(error))
try:
    from .score_mxnet import ScoreMxnet
except ImportError as error:
    ScoreMxnet = use_dummy('score_mxnet', error_message=str(error))
try:
    from .score_mxnet import ScoreMxnetV2
except ImportError as error:
    ScoreMxnetV2 = use_dummy('score_mxnet_v2', error_message=str(error))
try:
    from .score_tf import ScoreTensorFlow
except ImportError as error:
    ScoreTensorFlow = use_dummy('score_tf', error_message=str(error))
try:
    from .score_tf_v2 import ScoreTensorFlow
except ImportError as error:
    ScoreTensorFlow = use_dummy('score_tf_2x', error_message=str(error))
try:
    from .score_pytorch import PytorchPretrainedRunner
except ImportError as error:
    PytorchPretrainedRunner = use_dummy('score_pytorch_pretrained', error_message=str(error))
try:
    from .score_pytorch import PytorchTorchvisionRunner
except ImportError as error:
    PytorchTorchvisionRunner = use_dummy('score_pytorch_torchvision', error_message=str(error))
try:
    from .score_pytorch import PytorchTimmRunner
except ImportError as error:
    PytorchTimmRunner = use_dummy('score_pytorch_timm', error_message=str(error))
try:
    from .score_pytorch import PytorchTorchvisionOpticalFlowRunner
except ImportError as error:
    PytorchTorchvisionOpticalFlowRunner = use_dummy('score_pytorch_torchvision_optical_flow', error_message=str(error))
try:
    from .score_pytorch import PytorchTorchvisionDetectionRunner
except ImportError as error:
    PytorchTorchvisionDetectionRunner = use_dummy('score_pytorch_torchvision_detection', error_message=str(error))
try:
    from .score_pytorch import PytorchSavedModelRunner
except ImportError as error:
    PytorchSavedModelRunner = use_dummy('score_pytorch_saved_model', error_message=str(error))
try:
    from .score_pytorch_onnx_runtime import PytorchPretrainedToONNXRunner
except ImportError as error:
    PytorchPretrainedToONNXRunner = use_dummy('score_pytorch_pretrained_with_onnx', error_message=str(error))
try:
    from .score_pytorch_onnx_runtime import PytorchTorchvisionToONNXRunner
except ImportError as error:
    PytorchTorchvisionToONNXRunner = use_dummy('score_pytorch_torchvision_with_onnx', error_message=str(error))
try:
    from .score_caffe2 import Caffe2Runner
except ImportError as error:
    Caffe2Runner = use_dummy('score_caffe2', error_message=str(error))
try:
    from .score_onnx_runtime import ONNXRuntimeRunner
except ImportError as error:
    ONNXRuntimeRunner = use_dummy('score_onnx_runtime', error_message=str(error))
try:
    from .score_tf_lite import ScoreTensorFLowLite
except ImportError as error:
    ScoreTensorFLowLite = use_dummy('score_tf_lite', error_message=str(error))
try:
    from .score_tf_hub import ScoreTFHub
except ImportError as error:
    ScoreTensorFLowLite = use_dummy('score_tf_hub', error_message=str(error))
try:
    from .score_onnx_runtime import ONNXRuntimeRunner
except ImportError as error:
    ScoreTensorFLowLite = use_dummy('score_onnx_runtime', error_message=str(error))
try:
    from .score_kaldi import ScoreKaldi
except ImportError as error:
    ScoreKaldi = use_dummy('score_kaldi', error_message=str(error))
try:
    from .score_paddlepaddle import ScorePaddlePaddle
except ImportError as error:
    ScorePaddlePaddle = use_dummy('score_paddlepaddle', error_message=str(error))
