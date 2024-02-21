from .tf_hub_provider import ClassProvider
import numpy as np
import logging as log


class AssignIndicesTFHub(ClassProvider):
    """Assigns indices for tensors"""
    __action_name__ = "assign_indices_tf_hub"

    def __init__(self, config):
        pass

    @staticmethod
    def apply(data):
        if isinstance(data, np.ndarray):
            data = [data]
        converted = {}
        for i in range(len(data)):
            converted[i] = data[i]
        return converted


class AlignWithBatchTFHub(ClassProvider):
    """Batch alignment preprocessor.

    Aligns batch dimension in input data
    with BATCH value specified in test.

    Models 0-th dimension for batch and
    duplicates input data while size of batch
    dimension in input data won't be equal with BATCH.
    """
    __action_name__ = "align_with_batch_tf_hub"

    def __init__(self, config):
        self.batch = config["batch"]
        self.batch_dim = config.get('batch_dim', 0)
        self.expand_dims = config.get('expand_dims', True)
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Apply batch alignment (duplication) to data."""
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Align batch data for layers {} to batch {} ...".format(', '.join(
            '"{}"'.format(l) for l in apply_to), self.batch))
        for layer in apply_to:
            if self.expand_dims:
                data[layer] = np.expand_dims(data[layer], axis=self.batch_dim)
            data[layer] = np.repeat(data[layer], self.batch, axis=self.batch_dim)
        return data
