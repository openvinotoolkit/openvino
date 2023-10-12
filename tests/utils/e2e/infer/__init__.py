from .dummy_infer_class import use_dummy
from .provider import StepProvider

try:
    from .common_inference import Infer, AsyncInfer
except ImportError as e:
    Infer = use_dummy('ie_sync', str(e))
    AsyncInfer = use_dummy('ie_async', str(e))

try:
    from .reshape_no_infer import NoInfer
except ImportError as e:
    NoInfer = use_dummy('ie_no_infer', str(e))

try:
    from .infer_2_consecutive_reshape import Infer2ConsecutiveReshape
except ImportError as e:
    Infer2ConsecutiveReshape = use_dummy('ie_2_consecutive_reshape', str(e))

try:
    from .infer_speech_kaldi import SpeechKaldiInfer
except ImportError as e:
    SpeechKaldiInfer = use_dummy('ie_speech_kaldi', str(e))
