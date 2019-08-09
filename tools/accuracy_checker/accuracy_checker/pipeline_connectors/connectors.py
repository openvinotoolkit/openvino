from collections import namedtuple
from ..dependency import ClassProvider
from ..data_readers import DataRepresentation


StageConnectionDescription = namedtuple('StageConnection', ['from_stage', 'to_stage', 'replace', 'connector'])


class Connection:
    def __init__(self, stages, description: StageConnectionDescription):
        from_stage = description.from_stage
        if from_stage is None:
            for stage_index, stage in enumerate(stages):
                if stage == description.to_stage:
                    from_stage = list(stages.keys())[stage_index - 1]
        self.from_stage_context = stages[from_stage].evaluation_context
        self.to_stage_context = stages[description.to_stage].evaluation_context
        self.replace_container = description.replace
        if description.connector:
            self.connector = BaseConnector.provide(description.connector)
            self.replace_container = self.connector.replace_container

    def __call__(self, *args, **kwargs):
        shared_data = (
            self.connector(self.from_stage_context)
            if self.connector else getattr(self.from_stage_context, self.replace_container)
        )
        setattr(self.to_stage_context, self.replace_container, shared_data)


class BaseConnector(ClassProvider):
    __provider_type__ = 'connector'

    def connect(self, context):
        raise NotImplementedError

    def __call__(self, context, *args, **kwargs):
        return self.connect(context)


class PredictionToDataConnector(BaseConnector):
    __provider__ = 'prediction_to_data'

    replace_container = 'data_batch'

    def connect(self, context):
        batch_predictions = context.prediction_batch
        batch_identifiers = context.identifiers_batch
        data_batch = []
        for prediction_item, identifier in zip(batch_predictions, batch_identifiers):
            prediction_key = list(prediction_item.keys())[0]
            data_batch.append(DataRepresentation(prediction_item[prediction_key], identifier=identifier))

        return data_batch


def create_connection_description(configuration, stage_name):
    config = configuration
    if not isinstance(configuration, list):
        config = [configuration]
    for config_item in config:
        connector = config_item.get('connector')
        if connector:
            connected_stage = config_item.get('stage')
            return StageConnectionDescription(
                from_stage=connected_stage, to_stage=stage_name, replace=None, connector=connector
            )

    return None
