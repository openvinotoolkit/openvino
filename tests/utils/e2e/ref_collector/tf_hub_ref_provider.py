import inspect

from utils.e2e.common.base_provider import BaseProvider, BaseStepProvider


class ClassProvider(BaseProvider):
    registry = {}

    @classmethod
    def validate(cls):
        methods = [
            f[0] for f in inspect.getmembers(cls, predicate=inspect.isfunction)
        ]
        if 'get_refs' not in methods:
            raise AttributeError(
                "Requested class {} registred as '{}' doesn't provide required method get_refs"
                .format(cls.__name__, cls.__action_name__))


class TFHubStepProvider(BaseStepProvider):
    __step_name__ = "get_refs_tf_hub"

    def __init__(self, config):
        self.out_data = None
        action_name = next(iter(config))
        cfg = config[action_name]
        self.executor = ClassProvider.provide(action_name, config=cfg)

    def execute(self, data):
        self.out_data = self.executor.get_refs(*data)
