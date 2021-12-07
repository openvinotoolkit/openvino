import os
import functools
import numpy as np
import paddle
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import Quant2Int8MkldnnPass
from paddle.fluid import core

# ref: https://github.com/PaddlePaddle/PaddleSlim/blob/25f89072a280f8fdc63c7cd25fa54526dcbd9c51/paddleslim/quant/quanter.py#L397
def quant_post_static(
        executor,
        model_dir,
        quantize_model_path,
        batch_generator=None,
        sample_generator=None,
        model_filename=None,
        params_filename=None,
        save_model_filename='__model__',
        save_params_filename='__params__',
        batch_size=16,
        batch_nums=None,
        scope=None,
        algo='hist',
        hist_percent=0.9999,
        bias_correction=False,
        quantizable_op_type=["conv2d", "conv2d_transpose", "depthwise_conv2d", "matmul_v2", "pool2d", "mul"],
        is_full_quantize=False,
        weight_bits=8,
        activation_bits=8,
        activation_quantize_type='range_abs_max',
        weight_quantize_type='channel_wise_abs_max',
        optimize_model=False):
    post_training_quantization = PostTrainingQuantization(
        executor=executor,
        sample_generator=sample_generator,
        batch_generator=batch_generator,
        model_dir=model_dir,
        model_filename=model_filename,
        params_filename=params_filename,
        batch_size=batch_size,
        batch_nums=batch_nums,
        scope=scope,
        algo=algo,
        hist_percent=hist_percent,
        bias_correction=bias_correction,
        quantizable_op_type=quantizable_op_type,
        is_full_quantize=is_full_quantize,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        activation_quantize_type=activation_quantize_type,
        weight_quantize_type=weight_quantize_type,
        optimize_model=optimize_model)
    post_training_quantization.quantize()
    post_training_quantization.save_quantized_model(
        quantize_model_path,
        model_filename=save_model_filename,
        params_filename=save_params_filename)

fake_data = [-127, 127]
def get_x(shape):
    tmp = fake_data
    count = functools.reduce(lambda a, b: a * b, shape)
    l = tmp * (count // len(tmp) + 1)
    l = l[:count]
    n = np.array(l).astype(np.float32)
    n = np.reshape(n, shape)
    return n

def get_reader(shape, num):
    x = get_x(shape)

    def reader():
        for i in range(0, num):
            yield [x,]
    return reader

# ref: https://github.com/PaddlePaddle/PaddleSlim/blob/25f89072a280f8fdc63c7cd25fa54526dcbd9c51/demo/quant/quant_post/quant_post.py#L36
def quantize(config:dict):
    '''
    wrapper
    Args:
        config(dict):
            'batch_size', int,  32, "Minibatch size."
            'batch_num', int,  1, "Batch number"
            'model_path', str, "./inference_model/MobileNet/", "model dir"
            'save_path', str, "./quant_model/MobileNet/", "model dir to save quanted model"
            'model_filename', str, None, "model file name"
            'params_filename', str, None, "params file name"
            'algo', str, 'hist', "calibration algorithm"
            'hist_percent', float, 0.9999, "The percentile of algo:hist"
            'bias_correction', bool, False, "Whether to use bias correction")the model and params that saved by ``paddle.static.io.save_inference_model`` 
                are under the path.
            'shape',list,None,"shape"
            'num',int,None,"data number"
    '''
    place = paddle.CPUPlace()

    assert os.path.exists(config['model_path']), "config.model_path doesn't exist"
    assert os.path.isdir(config['model_path']), "config.model_path must be a dir"
    reader = get_reader(config['shape'], config['num'])

    exe = paddle.static.Executor(place)
    quant_post_static(
        executor=exe,
        model_dir=config['model_path'],
        quantize_model_path=config['save_path'],
        sample_generator=reader,
        model_filename=config['model_filename'],
        params_filename=config['params_filename'],
        batch_size=config['batch_size'],
        batch_nums=config['batch_num'],
        algo=config['algo'],
        hist_percent=config['hist_percent'],
        bias_correction=config['bias_correction'],
        save_model_filename=config['save_model_filename'],
        save_params_filename=config['save_params_filename'],
        #is_full_quantize=True,
        activation_quantize_type=config['activation_quantize_type'],
        weight_quantize_type=config['weight_quantize_type']
        )

# ref: https://github.com/PaddlePaddle/Paddle/blob/release/2.0/python/paddle/fluid/contrib/slim/tests/save_quant_model.py
def transform_and_save_int8_model(original_path, org_model_name, org_param_name, save_path, new_model_name, new_param_name):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.executor.global_scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
            fetch_targets] = fluid.io.load_inference_model(original_path, exe,
                                                        org_model_name, org_param_name)

        ops_to_quantize = set()
        op_ids_to_skip = set([-1])

        graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
        transform_to_mkldnn_int8_pass = Quant2Int8MkldnnPass(
            ops_to_quantize,
            _op_ids_to_skip=op_ids_to_skip,
            _scope=inference_scope,
            _place=place,
            _core=core)
        graph = transform_to_mkldnn_int8_pass.apply(graph)
        inference_program = graph.to_program()
        with fluid.scope_guard(inference_scope):
            fluid.io.save_inference_model(save_path, feed_target_names,
                                          fetch_targets, exe, inference_program,
                                          model_filename=new_model_name, params_filename=new_param_name)
        print(
            "Success! INT8 model obtained from the Quant model can be found at {}\n"
            .format(save_path))
