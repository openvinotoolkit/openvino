import os
import sys
import logging as log
import tensorflow as tf
from .tf_hub_ref_provider import ClassProvider


os.environ['GLOG_minloglevel'] = '3'


class ScoreTFHub(ClassProvider):
    """Reference collector for TensorFlow Hub models."""
    __action_name__ = "score_tf_hub"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self.res = {}

    def get_refs(self, input_data, model_obj):
        tf_inputs = {}
        for input_ind, input_name in enumerate(sorted(model_obj.structured_input_signature[1].keys())):
            tf_inputs[input_name] = tf.constant(input_data[input_ind])

        output_dict = {}
        for out_name, out_value in model_obj(**tf_inputs).items():
            output_dict[out_name] = out_value.numpy()

        for output_ind, external_name in enumerate(sorted(model_obj.structured_outputs.keys())):
            internal_name = model_obj.outputs[output_ind].name
            out_value = output_dict[external_name]
            self.res[internal_name] = out_value

        return self.res

