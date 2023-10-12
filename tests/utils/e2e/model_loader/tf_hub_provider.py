import inspect

from tests.utils.e2e.common.base_provider import BaseProvider, BaseStepProvider


class ClassProvider(BaseProvider):
    registry = {}

    @classmethod
    def validate(cls):
        methods = [
            f[0] for f in inspect.getmembers(cls, predicate=inspect.isfunction)
        ]
        if 'load_model' not in methods:
            raise AttributeError(
                "Requested class {} registered as '{}' doesn't provide required method load_model"
                .format(cls.__name__, cls.__action_name__))


class TFHubStepProvider(BaseStepProvider):
    __step_name__ = "tf_hub_load_model"

    def __init__(self, config):
        self.model = None
        self.out_data = None
        action_name = next(iter(config))
        cfg = config[action_name]
        self.executor = ClassProvider.provide(action_name, config=cfg)

    def execute(self, data=None):
        self.out_data = self.executor.load_model(data)
