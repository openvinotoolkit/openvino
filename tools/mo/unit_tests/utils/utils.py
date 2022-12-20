import argparse
import os


def base_args_config():
    args = argparse.Namespace()
    args.extensions = [os.getcwd()]
    args.use_legacy_frontend = False
    args.use_new_frontend = False
    args.framework = 'tf'
    args.model_name = None
    args.input_model = None
    args.silent = True
    args.transform = []
    args.scale = None
    args.output = None
    args.input = None
    args.input_shape = None
    args.batch = None
    args.mean_values = ()
    args.scale_values = ()
    args.output_dir = os.getcwd()
    args.freeze_placeholder_with_value = None
    args.transformations_config = None
    args.disable_fusing = None
    args.finegrain_fusing = None
    args.disable_resnet_optimization = None
    args.enable_concat_optimization = None
    args.static_shape = None
    args.disable_weights_compression = None
    args.reverse_input_channels = None
    args.data_type = None
    args.layout = ()
    args.source_layout = ()
    args.target_layout = ()
    args.input_checkpoint = None
    args.saved_model_dir = None
    args.input_meta_graph = None
    args.saved_model_tags = None
    args.progress = True
    args.stream_output = False
    args.tensorflow_use_custom_operations_config = None
    args.tensorflow_custom_layer_libraries = None
    args.tensorflow_custom_operations_config_update = None
    args.tensorboard_logdir = None
    args.disable_nhwc_to_nchw = False
    return args
