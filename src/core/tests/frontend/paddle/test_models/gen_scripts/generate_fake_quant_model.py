import numpy as np
import sys
import paddle as pdpd
from quant.util import quantize, fake_data
import paddle.fluid as fluid
import os
import save_model
try:
    # paddle version=2.2.1
    from paddle.nn.quant.quant_layers import FakeQuantMovingAverageAbsMax
    from paddle.nn.quant.quant_layers import FakeQuantAbsMax
    from paddle.nn.quant.quant_layers import FakeQuantChannelWiseAbsMax
except:
    # paddle version=2.1.0
    from paddle.fluid.contrib.slim.quantization.imperative.quant_nn import FakeQuantMovingAverage as FakeQuantMovingAverageAbsMax
    from paddle.fluid.contrib.slim.quantization.imperative.quant_nn import FakeQuantAbsMax
    from paddle.fluid.contrib.slim.quantization.imperative.quant_nn import FakeChannelWiseQuantDequantAbsMax as FakeQuantChannelWiseAbsMax

def saveModel(name, exe, feedkeys:list, fetchlist:list, inputs:list, outputs:list, target_dir:str):
    model_dir = os.path.join(target_dir, name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for i, input in enumerate(inputs):
        np.save(os.path.join(model_dir, "input{}".format(i)), input)
        np.save(os.path.join(model_dir, "input{}.{}.{}".format(i, feedkeys[i], input.dtype)), input)

    for i, output in enumerate(outputs):
        np.save(os.path.join(model_dir, "output{}".format(i)), output)     

def test_conv(name, model_type, x, data_alg, weight_alg, groups):
    pdpd.enable_static()
    a = np.array([[[[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]]]]).astype(np.float32)
    if model_type == 'conv2d_transpose':
        if groups == 1:
            kernel = np.zeros((int(x.shape[1]), int(x.shape[1]), 3, 3)).astype(np.float32)
        else:
            kernel = np.zeros((int(x.shape[1]), int(4 / groups), 3, 3)).astype(np.float32)
    else:
        kernel = np.zeros((2, int(x.shape[1] / groups), 3, 3)).astype(np.float32)
    kernel[:,:] = a
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        weight_attr = pdpd.ParamAttr(name="conv2d_weight", initializer=pdpd.nn.initializer.Assign(kernel))
        if model_type == 'conv2d_transpose':
            conv = pdpd.static.nn.conv2d_transpose(input=node_x, num_filters=kernel.shape[0], filter_size=kernel.shape[2:4],
                                        param_attr=weight_attr, bias_attr=False,
                                        groups=groups)
        else:
            conv = pdpd.static.nn.conv2d(input=node_x, num_filters=kernel.shape[0], filter_size=kernel.shape[2:4],
                                        param_attr=weight_attr, bias_attr=False,
                                        groups=groups)
        result = pdpd.fluid.layers.cast(conv, np.float32)
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        model_dir = os.path.join(sys.argv[1], name)
        feedkeys = ['x']
        fetchlist = [result]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pdpd.fluid.io.save_inference_model(model_dir, feedkeys, fetchlist, exe, model_filename="_x.pdmodel", params_filename="_x.pdiparams")

        config = {
            'batch_size': 1,                        # "Minibatch size."
            'batch_num': 1,                         # "Batch number"
            'model_path': model_dir,                # "model dir"
            'save_path': model_dir,                 # "model dir to save quanted model"
            'model_filename': '_x.pdmodel',         # "model file name"
            'params_filename': "_x.pdiparams",      # "params file name"
            'algo': 'abs_max',                      # "calibration algorithm"
            'hist_percent': 0.9999,                 # "The percentile of algo:hist"
            'bias_correction': False,               # "Whether to use bias correction")the model and params that saved by ``paddle.static.io.save_inference_model`` 
                                                    # are under the path.
            'shape': x.shape,                       # "shape"
            'num': 1,                               # "data number"
            'save_model_filename': f"{model_dir}/{name}.pdmodel",
            'save_params_filename': f"{model_dir}/{name}.pdiparams",
            'activation_quantize_type': data_alg,
                                                    # range_abs_max moving_average_abs_max abs_max(?)
            'weight_quantize_type': weight_alg,
                                                    # channel_wise_abs_max abs_max
        }
        quantize(config)
        # save int8 model
        # transform_and_save_int8_model(model_dir, config["save_model_filename"], config["save_params_filename"], model_dir, '_x_int8.pdmodel', '_x_int8.pdiparams')
        # [inference_program, feed_target_names, fetch_targets] = pdpd.fluid.io.load_inference_model(model_dir, exe, f'_x_int8.pdmodel', f'_x_int8.pdiparams')
        # outs = exe.run(inference_program, feed={'x': x}, fetch_list=fetch_targets)
        # print('int8:', outs[0])
        [inference_program, feed_target_names, fetch_targets] = pdpd.fluid.io.load_inference_model(model_dir, exe, config["save_model_filename"], config["save_params_filename"])
        outs = exe.run(inference_program, feed={'x': x}, fetch_list=fetch_targets)
        print('float', outs[0])
        saveModel(name, exe, feedkeys=feed_target_names, fetchlist=fetch_targets, inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])
    return outs[0]

def test_matmul(name, x, w, data_alg, weight_alg, op_type='matmul'):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x1 = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_x2 = pdpd.fluid.layers.create_parameter(w.shape, w.dtype, 'weight', 
            default_initializer=pdpd.nn.initializer.Assign(w))
        if op_type == 'matmul':
            mat = pdpd.fluid.layers.matmul(node_x1, node_x2, False, False)
        else:
            mat = pdpd.fluid.layers.mul(node_x1, node_x2)
        result = pdpd.fluid.layers.cast(mat, np.float32)
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        model_dir = os.path.join(sys.argv[1], name)
        feedkeys = ['x']
        fetchlist = [result]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pdpd.fluid.io.save_inference_model(model_dir, feedkeys, fetchlist, exe, model_filename="_x.pdmodel", params_filename="_x.pdiparams")

        config = {
            'batch_size': 1,                        # "Minibatch size."
            'batch_num': 1,                         # "Batch number"
            'model_path': model_dir,                # "model dir"
            'save_path': model_dir,                 # "model dir to save quanted model"
            'model_filename': '_x.pdmodel',         # "model file name"
            'params_filename': '_x.pdiparams',      # "params file name"
            'algo': 'abs_max',                      # "calibration algorithm"
            'hist_percent': 0.9999,                 # "The percentile of algo:hist"
            'bias_correction': False,               # "Whether to use bias correction")the model and params that saved by ``paddle.static.io.save_inference_model`` 
                                                    # are under the path.
            'shape': x.shape,                      # "shape"
            'num': 1,                               # "data number"
            'save_model_filename': f"{model_dir}/{name}.pdmodel",
            'save_params_filename': f"{model_dir}/{name}.pdiparams",
            'activation_quantize_type': data_alg,
                                                    # range_abs_max moving_average_abs_max abs_max(?)
            'weight_quantize_type': weight_alg,
                                                    # channel_wise_abs_max abs_max
        }
        quantize(config)
        # save int8 model
        # transform_and_save_int8_model(model_dir, config["save_model_filename"], config["save_params_filename"], model_dir, '_x_int8.pdmodel', '_x_int8.pdiparams')
        # [inference_program, feed_target_names, fetch_targets] = pdpd.fluid.io.load_inference_model(model_dir, exe, f'_x_int8.pdmodel', f'_x_int8.pdiparams')
        # outs = exe.run(inference_program, feed={'x': x}, fetch_list=fetch_targets)
        # print('int8:', outs[0])
        [inference_program, feed_target_names, fetch_targets] = pdpd.fluid.io.load_inference_model(model_dir, exe, config["save_model_filename"], config["save_params_filename"])
        outs = exe.run(inference_program, feed={'x': x}, fetch_list=fetch_targets)
        print('float', outs[0])
        saveModel(name, exe, feedkeys=feed_target_names, fetchlist=fetch_targets, inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])
    return outs[0]

def test_pool2d(name, x, data_alg):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x1 = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        mat = pdpd.fluid.layers.pool2d(node_x1, pool_size=2)
        result = pdpd.fluid.layers.cast(mat, np.float32)
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        model_dir = os.path.join(sys.argv[1], name)
        feedkeys = ['x']
        fetchlist = [result]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pdpd.fluid.io.save_inference_model(model_dir, feedkeys, fetchlist, exe, model_filename="_x.pdmodel", params_filename="_x.pdiparams")

        config = {
            'batch_size': 1,                        # "Minibatch size."
            'batch_num': 1,                         # "Batch number"
            'model_path': model_dir,                # "model dir"
            'save_path': model_dir,                 # "model dir to save quanted model"
            'model_filename': '_x.pdmodel',         # "model file name"
            'params_filename': None,                # "params file name"
            'algo': 'abs_max',                      # "calibration algorithm"
            'hist_percent': 0.9999,                 # "The percentile of algo:hist"
            'bias_correction': False,               # "Whether to use bias correction")the model and params that saved by ``paddle.static.io.save_inference_model`` 
                                                    # are under the path.
            'shape': x.shape,                       # "shape"
            'num': 1,                               # "data number"
            'save_model_filename': f"{model_dir}/{name}.pdmodel",
            'save_params_filename': f"{model_dir}/{name}.pdiparams",
            'activation_quantize_type': data_alg,
                                                    # range_abs_max moving_average_abs_max abs_max(?)
            'weight_quantize_type': 'abs_max',
                                                    # channel_wise_abs_max abs_max
        }
        quantize(config)
        # save int8 model
        # transform_and_save_int8_model(model_dir, config["save_model_filename"], config["save_params_filename"], model_dir, '_x_int8.pdmodel', '_x_int8.pdiparams')
        # [inference_program, feed_target_names, fetch_targets] = pdpd.fluid.io.load_inference_model(model_dir, exe, f'_x_int8.pdmodel', f'_x_int8.pdiparams')
        # outs = exe.run(inference_program, feed={'x': x}, fetch_list=fetch_targets)
        # print('int8:', outs[0])
        [inference_program, feed_target_names, fetch_targets] = pdpd.fluid.io.load_inference_model(model_dir, exe, config["save_model_filename"], config["save_params_filename"])
        outs = exe.run(inference_program, feed={'x': x}, fetch_list=fetch_targets)
        print('float', outs[0])
        saveModel(name, exe, feedkeys=feed_target_names, fetchlist=fetch_targets, inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])
    return outs[0]

# PostTrainingQuantization could not generate fake_quantize_dequantize_abs_max and 
#    fake_channel_wise_quantize_dequantize_abs_max, make a network directly with them
def test_matmul_quant_dequant(name, weight_alg, x, w):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x1 = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_x2 = pdpd.fluid.layers.create_parameter(w.shape, w.dtype, 'weight', 
            default_initializer=pdpd.nn.initializer.Assign(w))
        act_scale = FakeQuantMovingAverageAbsMax(name=node_x1.name)
        # set the scale
        scale_attr = pdpd.ParamAttr(
            name='act_data',
            initializer=pdpd.fluid.initializer.Constant(max(fake_data)),
            trainable=False)
        act_scale._scale = pdpd.fluid.layers.create_parameter(
            shape=[1], attr=scale_attr, dtype='float32')
        fake_op_act = act_scale(node_x1)
        if weight_alg == 'abs_max':
            weight_scale = FakeQuantAbsMax(name=node_x2.name, quant_bits=8, dtype=node_x2.dtype, quant_on_weight=True)
        else:
            weight_scale = FakeQuantChannelWiseAbsMax(name=node_x2.name, channel_num=1, quant_axis=1, quant_bits=8, dtype=node_x2.dtype, quant_on_weight=True)
        fake_op_weight = weight_scale(node_x2)
        mat = pdpd.fluid.layers.matmul(fake_op_act, fake_op_weight, False, False)
        result = pdpd.fluid.layers.cast(mat, np.float32)
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(feed={'x': x}, fetch_list=[result])
        save_model.saveModel(name, exe, feedkeys=['x'], fetchlist=[result], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])
    return outs[0]


def get_x(shape):
    import functools
    tmp = [-100, -120, -110, -125,
           -100, -120, -110, -125,
           -100, -120, -110, -125,
           -100, -120, -110, -125
           ]
    count = functools.reduce(lambda a, b: a * b, shape)
    l = tmp * (count // len(tmp) + 1)
    l = l[:count]
    n = np.array(l).astype(np.float32)
    x = np.reshape(n, shape)
    return x

if __name__ == "__main__":
    x = get_x((1, 8, 4, 4))
    # fake_quantize_range_abs_max->conv2d->fake_dequantize_max_abs
    test_conv('fake_conv2d_range_abs_max+abs_max', 'conv2d', x, 'range_abs_max', 'abs_max', 1)
    # fake_quantize_moving_average_abs_max->conv2d->fake_channel_wise_dequantize_max_abs
    test_conv('fake_conv2d_moving_average_abs_max+channel_wise_abs_max', 'conv2d', x, 'moving_average_abs_max', 'channel_wise_abs_max', 1)
    # fake_quantize_moving_average_abs_max->(group)conv2d->fake_channel_wise_dequantize_max_abs
    test_conv('fake_depthwise_conv2d_moving_average_abs_max+channel_wise_abs_max', 'depthwise_conv2d', x, 'moving_average_abs_max', 'channel_wise_abs_max', 2)
    # fake_quantize_range_abs_max->conv2d_transpose->fake_dequantize_max_abs
    test_conv('fake_conv2d_transpose_range_abs_max+abs_max', 'conv2d_transpose', x, 'range_abs_max', 'abs_max', 1)
    #x = get_x((1, 4, 4, 4)) # quantized model cannot be executed
    # fake_quantize_range_abs_max->(group)conv2d_transpose->fake_dequantize_max_abs
    #test_conv('fake_conv2d_transpose_range_abs_max+channel_wise_abs_max_group', 'conv2d_transpose', x, 'range_abs_max', 'channel_wise_abs_max', 2)
    x = np.round(get_x((1, 1, 4, 4)) / 10).astype(np.float32)
    w = np.array([[[[2, -17, -3, 21],
                   [4, -38, 3, -20],
                   [-4, 11, -2, 15],
                   [-2, 45, 2, -14]]]]).astype(np.float32)
    # fake_quantize_moving_average_abs_max->matmul->fake_dequantize_max_abs
    test_matmul('fake_matmul_moving_average_abs_max+abs_max', x, w, 'moving_average_abs_max', 'abs_max')
    # fake_quantize_range_abs_max->matmul->fake_dequantize_max_abs
    test_matmul('fake_matmul_range_abs_max+abs_max', x, w, 'range_abs_max', 'abs_max')
    # data->fake_quantize_dequantize_moving_average_abs_max->matmul
    # weight->fake_quantize_dequantize_abs_max             ->
    test_matmul_quant_dequant('fake_matmul_quantize_dequantize_abs_max', 'abs_max', x, w)
    x = np.round(get_x((4, 4)) / 10).astype(np.float32)
    w = np.array([[2, -17, -3, 21],
                   [4, -38, 3, -20],
                   [-4, 11, -2, 15],
                   [-2, 45, 2, -14]]).astype(np.float32)
    # fake_quantize_moving_average_abs_max->mul->fake_dequantize_max_abs
    test_matmul('fake_mul_moving_average_abs_max+abs_max', x, w, 'moving_average_abs_max', 'abs_max', 'mul')
    x = get_x((1, 8, 4, 4))
    # fake_quantize_dequantize_moving_average_abs_max->pool2d
    test_pool2d('fake_pool2d_moving_average_abs_max', x, 'moving_average_abs_max')
