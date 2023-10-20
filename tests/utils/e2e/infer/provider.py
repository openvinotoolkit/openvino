import inspect
from e2e_oss.utils.test_utils import log_timestamp
from utils.e2e.common.base_provider import BaseProvider, BaseStepProvider


class ClassProvider(BaseProvider):
    registry = {}

    @classmethod
    def validate(cls):
        methods = [
            f[0] for f in inspect.getmembers(cls, predicate=inspect.isfunction)
        ]
        if 'infer' not in methods:
            raise AttributeError(
                "Requested class {} registred as '{}' doesn't provide required method infer"
                .format(cls.__name__, cls.__action_name__))


class StepProvider(BaseStepProvider):
    __step_name__ = "infer"

    def __init__(self, config):
        self.out_data = None
        action_name = next(iter(config))
        self.executor = ClassProvider.provide(action_name, config=config[action_name])

    def execute(self, data):
        with log_timestamp('Inference'):
            self.out_data = self.executor.infer(data)
