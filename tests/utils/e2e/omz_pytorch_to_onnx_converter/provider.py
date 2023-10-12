import inspect
from tests.utils.e2e.common.base_provider import BaseProvider, BaseStepProvider


class ClassProvider(BaseProvider):
    registry = {}

    @classmethod
    def validate(cls):
        methods = [
            f[0] for f in inspect.getmembers(cls, predicate=inspect.isfunction)
        ]
        if 'convert_pytorch_to_onnx' not in methods:
            raise AttributeError(
                "Requested class {} registred as '{}' doesn't provide required method 'convert'"
                .format(cls.__name__, cls.__action_name__))


class StepProvider(BaseStepProvider):
    __step_name__ = "pytorch_to_onnx"

    def __init__(self, config):
        self.out_data = None
        action_name = next(iter(config))
        cfg = config[action_name]
        self.executor = ClassProvider.provide(action_name, config=cfg)

    def execute(self, data=None):
        self.out_data = data
        self.executor.convert_pytorch_to_onnx(data)
