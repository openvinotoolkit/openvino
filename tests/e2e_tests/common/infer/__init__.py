from .dummy_infer_class import use_dummy
from .provider import StepProvider

try:
    from .common_inference import Infer
except ImportError as e:
    Infer = use_dummy('ie_sync', str(e))
