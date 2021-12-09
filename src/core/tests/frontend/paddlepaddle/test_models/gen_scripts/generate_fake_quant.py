import numpy as np
from save_model import saveModel
import sys
import paddle as pdpd
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type

# ref: https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_fake_quantize_op.py
def fake_quantize_abs_max(x, scale, name=None):
    attrs = {'bit_length': 8}
    op_type = 'fake_quantize_abs_max'
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], op_type)

    helper = LayerHelper(op_type, **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)
    helper.append_op(
        type=op_type,
        inputs={'X': x},
        outputs={'Out': out,
            'OutScale': scale
            },
        attrs=attrs)
    return out

def fake_channel_wise_quantize_abs_max(x, scale, quant_axis=0, name=None):
    attrs = {'bit_length': 8, 'quant_axis': quant_axis}
    op_type = 'fake_channel_wise_quantize_abs_max'
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], op_type)

    helper = LayerHelper(op_type, **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)
    helper.append_op(
        type=op_type,
        inputs={'X': x},
        outputs={'Out': out,
            'OutScale': scale
            },
        attrs=attrs)
    return out

def fake_quantize_range_abs_max(x, scale_in, scale_out, scales, iter, window_size=1, name=None):
    attrs = {'bit_length': 8, 'window_size': window_size, 'is_test': True}
    op_type = 'fake_quantize_range_abs_max'
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], op_type)

    helper = LayerHelper(op_type, **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)
    helper.append_op(
        type=op_type,
        inputs={'X': x,
            'Iter': iter,
            'InScale': scale_in
            },
        outputs={'Out': out,
            'OutScale': scale_out,
            'OutScales': scales
            },
        attrs=attrs)
    return out

def fake_quantize_template(op_type, x, scale_in, scale_out, name=None):
    attrs = {'bit_length': 8, 'moving_rate': 0.9, 'is_test': True}
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], op_type)

    helper = LayerHelper(op_type, **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)
    helper.append_op(
        type=op_type,
        inputs={'X': x,
            'InScale': scale_in
            },
        outputs={'Out': out,
            'OutScale': scale_out
            },
        attrs=attrs)
    return out

def fake_quantize_dequantize_abs_max(x, scale_in, scale_out, name=None):
    attrs = {'bit_length': 8}
    op_type = 'fake_quantize_dequantize_abs_max'
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], op_type)

    helper = LayerHelper(op_type, **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)
    helper.append_op(
        type=op_type,
        inputs={'X': x
            },
        outputs={'Out': out,
            'OutScale': scale_out
            },
        attrs=attrs)
    return out

def fake_channel_wise_quantize_dequantize_abs_max(x, scale, quant_axis=0, name=None):
    attrs = {'bit_length': 8, 'quant_axis': quant_axis}
    op_type = 'fake_channel_wise_quantize_dequantize_abs_max'
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], op_type)

    helper = LayerHelper(op_type, **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)
    helper.append_op(
        type=op_type,
        inputs={'X': x},
        outputs={'Out': out,
            'OutScale': scale
            },
        attrs=attrs)
    return out

def fake_dequantize_max_abs(x, scale, name=None):
    attrs = {'max_range': 127}
    op_type = 'fake_dequantize_max_abs'
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], op_type)

    helper = LayerHelper(op_type, **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)
    helper.append_op(
        type=op_type,
        inputs={'X': x,
            'Scale': scale
            },
        outputs={'Out': out,
            },
        attrs=attrs)
    return out

def fake_channel_wise_dequantize_max_abs(x, scale, quant_axis=0, name=None):
    attrs = {'quant_bits': [8,], 'quant_axis': quant_axis}
    op_type = 'fake_channel_wise_dequantize_max_abs'
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], op_type)

    helper = LayerHelper(op_type, **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)
    helper.append_op(
        type=op_type,
        inputs={'X': x,
            'Scales':[
                scale,
            ]},
        outputs={'Out': out
            },
        attrs=attrs)
    return out

def test_fake_quantize_abs_max(name, x):
    scale = np.max(np.abs(x))
    pdpd.enable_static()
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_scale = pdpd.static.data(name='scale', shape=(), dtype=pdpd.float32)
        result = fake_quantize_abs_max(node_x, node_scale)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[result, node_scale])
        saveModel(name, exe, feedkeys=['x'], fetchlist=[result], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])
        assert(scale == outs[1])
    return outs[0]

def test_fake_channel_wise_quantize_abs_max(name, x, quant_axis):
    scales = []
    if quant_axis == 0:
        for i in range(x.shape[0]):
            scale_v = np.max(np.abs(x[i])).astype("float32")
            scales.append(scale_v)
    else:
        for i in range(x.shape[1]):
            scale_v = np.max(np.abs(x[:, i])).astype("float32")
            scales.append(scale_v)
    scalesp = np.zeros(len(scales)).astype("float32")
    pdpd.enable_static()
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_scale = pdpd.static.data(name='scale', shape=scalesp.shape, dtype=scalesp.dtype)
        result = fake_channel_wise_quantize_abs_max(node_x, node_scale, quant_axis)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[result, node_scale])
        assert(scales == list(outs[1]))
        saveModel(name, exe, feedkeys=['x'], fetchlist=[result], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])
    return outs[0]

def test_fake_quantize_range_abs_max(name, x):
    scale = np.array([np.max(np.abs(x)).astype("float32") - 1.0]).astype('float32')
    out_scales = np.zeros(1).astype("float32")
    out_scales[0] = scale.astype('float32')
    iter = np.zeros(1).astype("int64")
    pdpd.enable_static()
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_scale_in = pdpd.static.data(name='scale', shape=scale.shape, dtype=scale.dtype)
        node_scale_out = pdpd.static.data(name='scale_out', shape=scale.shape, dtype=scale.dtype)
        node_scales = pdpd.static.data(name='scales', shape=out_scales.shape, dtype=out_scales.dtype)
        node_iter = pdpd.static.data(name='iter', shape=iter.shape, dtype=iter.dtype)
        result = fake_quantize_range_abs_max(node_x, node_scale_in, node_scale_out, node_scales, node_iter)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x,
                'iter': iter,
                'scale': scale
                },
            fetch_list=[result, node_scale_out, node_scales])
        # TODO: could not get the scale in output
        #assert(scale == outs[1])
        #assert(scale == outs[2][0])
        saveModel(name, exe, feedkeys=['x'], fetchlist=[result], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])
    return outs[0]

def test_fake_template(name, op_type, x):
    scale_in = np.array([16,]).astype('float32')
    out_scale = np.zeros(1).astype('float32')
    out_scale[0] = (0.9 * 1 + np.max(np.abs(x))) / (0.9 * 1 + 1)
    pdpd.enable_static()
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_scale_in = pdpd.static.data(name='scale_in', shape=scale_in.shape, dtype=scale_in.dtype)
        node_scale_out = pdpd.static.data(name='scale_out', shape=out_scale.shape, dtype=out_scale.dtype)
        result = fake_quantize_template(op_type, node_x, node_scale_in, node_scale_out)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x,
                'scale_in': scale_in,
                },
            fetch_list=[result, node_scale_out])
        # TODO: cannot get scale in output
        #assert(outs[1] == out_scale[0])
        saveModel(name, exe, feedkeys=['x', 'scale_in'], fetchlist=[result], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])
    return outs[0]

def test_fake_quantize_dequantize_abs_max(name, x):
    scale_in = np.max(np.abs(x)).astype("float32")
    scale_out = np.array(scale_in).astype('float32')
    pdpd.enable_static()
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_scale_in = pdpd.static.data(name='scale_in', shape=scale_in.shape, dtype=scale_in.dtype)
        node_scale_out = pdpd.static.data(name='scale_out', shape=scale_out.shape, dtype=scale_out.dtype)
        result = fake_quantize_dequantize_abs_max(node_x, node_scale_in, node_scale_out)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x,
                },
            fetch_list=[result, node_scale_out])
        assert(outs[1] == scale_in)
        saveModel(name, exe, feedkeys=['x'], fetchlist=[result], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])
    return outs[0]

def test_fake_channel_wise_quantize_dequantize_abs_max(name, x, quant_axis):
    scales = []
    if quant_axis == 0:
        for i in range(x.shape[0]):
            scale_v = np.max(np.abs(x[i])).astype("float32")
            scales.append(scale_v)
    else:
        for i in range(x.shape[1]):
            scale_v = np.max(np.abs(x[:, i])).astype("float32")
            scales.append(scale_v)
    scales = np.array(scales).astype("float32")
    pdpd.enable_static()
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_scale = pdpd.static.data(name='scale', shape=scales.shape, dtype=scales.dtype)
        result = fake_channel_wise_quantize_dequantize_abs_max(node_x, node_scale, quant_axis)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[result, node_scale])
        assert(list(outs[1]) == list(scales))
        saveModel(name, exe, feedkeys=['x'], fetchlist=[result], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])
    return outs[0]

def test_fake_dequantize_max_abs(name, x):
    scale = np.zeros(1).astype("float32")
    scale[0] = np.max(np.abs(x))
    xq = np.round(x / scale * 127)  # quantize
    pdpd.enable_static()
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=xq.shape, dtype=xq.dtype)
        node_scale = pdpd.static.data(name='scale', shape=scale.shape, dtype=scale.dtype)
        result = fake_dequantize_max_abs(node_x, node_scale)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': xq, 'scale': scale},
            fetch_list=[result])
        saveModel(name, exe, feedkeys=['x', 'scale'], fetchlist=[result], inputs=[xq, scale], outputs=[outs[0]], target_dir=sys.argv[1])
    return outs[0]

def test_fake_channel_wise_dequantize_max_abs(name, x, quant_axis):
    scales = []
    xq = x.copy()
    if quant_axis == 0:
        for i in range(x.shape[0]):
            scale_v = np.max(np.abs(x[i])).astype("float32")
            xq[i] = np.round(x[i] * 127 / scale_v)
            scales.append(scale_v)
    else:
        for i in range(x.shape[1]):
            scale_v = np.max(np.abs(x[:, i])).astype("float32")
            xq[:, i] = np.round(x[:, i] * 127 / scale_v)
            scales.append(scale_v)
    scales = np.array(scales).astype("float32")
    pdpd.enable_static()
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=xq.shape, dtype=xq.dtype)
        node_scale = pdpd.static.data(name='scale', shape=scales.shape, dtype=scales.dtype)
        result = fake_channel_wise_dequantize_max_abs(node_x, node_scale, quant_axis)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': xq, 'scale': scales},
            fetch_list=[result])
        saveModel(name, exe, feedkeys=['x', 'scale'], fetchlist=[result], inputs=[xq, scales], outputs=[outs[0]], target_dir=sys.argv[1])
    return outs[0]

if __name__ == "__main__":
    # quant: scale is 'OutScale' in output, and it depends on x, paddle lite does not handle
    x = np.array((3.5, 7.0, -7.0)).astype('float32')
    test_fake_quantize_abs_max("fake_quantize_abs_max", x)
    x = np.array([[3.5/1, 7.0/1, -7.0/1],
                  [3.5/2, 7.0/2, -7.0/2],
                  [3.5*2, 7.0*2, -7.0*2]]).astype('float32')
    test_fake_channel_wise_quantize_abs_max('fake_channel_wise_quantize_abs_max_axis0', x, 0)
    test_fake_channel_wise_quantize_abs_max('fake_channel_wise_quantize_abs_max_axis1', x, 1)

    ## paddle lite handles fake_quantize_range_abs_max, fake_quantize_moving_average_abs_max in DeleteQuantOpFuser
    ## it gets the scale in 'OutScale'
    # quant: scale is 'InScale' in input
    x = (np.random.random((8, 16, 7, 7)) - 0.5) * 10
    x = x.astype("float32")
    test_fake_quantize_range_abs_max('fake_quantize_range_abs_max', x)
    # quant: scale is 'InScale' in input, no scale in output
    x = np.array((8, 4, -8)).astype('float32')
    op_type = 'fake_quantize_moving_average_abs_max'
    test_fake_template(op_type, op_type, x)
    ## lite handles 'fake_dequantize_max_abs' in DequantOpFuser
    ## it gets the scale in dequant op's attributes 'max_range', quant op's 'bit_length'
    # dequant: scale is 'Scale'/'Scales' in input
    x = np.random.randn(31, 65).astype('float32')
    test_fake_dequantize_max_abs('fake_dequantize_max_abs', x)
    ## lite handles 'fake_channel_wise_dequantize_max_abs' in ChannelWiseDequantOpFuser
    ## it gets the scale in dequant op's attributes 'quant_bits', dequant op's input
    x = np.random.randn(4, 3, 64, 64).astype('float32')
    test_fake_channel_wise_dequantize_max_abs('fake_channel_wise_dequantize_max_abs_axis0', x, 0)
    test_fake_channel_wise_dequantize_max_abs('fake_channel_wise_dequantize_max_abs_axis1', x, 1)

    ## lite handles in QuantDequantOpFuser
    ## when input is activation and it's fake_quantize_dequantize_moving_average_abs_max,
    ##   scale = 'OutScale'
    ## when input is weight and it's fake_quantize_dequantize_abs_max,
    ##   scale = max(abs(x))
    ## when input is weight and it's fake_channel_wise_quantize_dequantize_abs_max, ???
    ##   scale = 'OutScale'
    # quant&dequant: scale is 'InScale' in input, no scale in output
    op_type = 'fake_quantize_dequantize_moving_average_abs_max'
    test_fake_template(op_type, op_type, x)
    # quant&dequant: scale is 'OutScale' in output, no scale in input, and it depends on x
    test_fake_quantize_dequantize_abs_max('fake_quantize_dequantize_abs_max', x)
    # quant&dequant: scale is 'OutScale' in output, no scale in input, and it depends on x
    x = np.array([[8, 4, -8],[4,-8,8],[-8,8,4]]).astype('float32')
    test_fake_channel_wise_quantize_dequantize_abs_max('fake_channel_wise_quantize_dequantize_abs_max_axis0', x, 0)
    test_fake_channel_wise_quantize_dequantize_abs_max('fake_channel_wise_quantize_dequantize_abs_max_axis1', x, 1)
