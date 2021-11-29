#
# pool2d paddle model generator
#
import numpy as np
import sys
from typing import List

from save_model import saveModel


# helpers
def pdpd_attr_to_list(s, p, d):
    if isinstance(s, int):
        strides = [s, ]*2
    elif isinstance(s, tuple):
        strides = list(s)
    elif isinstance(s, list):
        strides = s
    else:
        raise ValueError('unknown type of strides!')
    assert len(strides) == 2, 'len(strides) must be 2!'

    if isinstance(d, int):
        dilations = [d, ]*2
    elif isinstance(d, tuple):
        dilations = list(d)
    elif isinstance(d, list):
        dilations = d
    else:
        raise ValueError('unknown type of dilations!')
    assert len(dilations) == 2, 'len(dilations) must be 2!'

    if isinstance(p, int):
        padding = [p, ]*2
    elif isinstance(p, tuple):
        p = list(p)
        padding = p  # PDPD: (pad_height, pad_width)
    elif isinstance(p, list):
        padding = p
    else:
        raise ValueError('unknown type of padding!')
    assert len(padding) == 2, 'len(padding) must be 2!'

    return strides, padding, dilations


def ngraph_deform_conv(test_x, weight, offset, mask, bias,
                       strides: List[int] = [1, 1],
                       pads_begin: List[int] = [0, 0],
                       pads_end: List[int] = [0, 0],
                       dilations: List[int] = [1, 1],
                       deformable_groups=1, groups=1):
    import ngraph as ng
    from ngraph import opset8 as opset
    from openvino.inference_engine import IECore

    node_x = ng.parameter(shape=test_x.shape, name='x', dtype=np.float32)
    node_deform_values = ng.parameter(
        shape=offset.shape, name='deform_values', dtype=np.float32)
    node_w = ng.parameter(shape=weight.shape, name='w', dtype=np.float32)

    graph = opset.deformable_convolution(data=node_x,
                                         offsets=node_deform_values,
                                         filters=node_w,
                                         strides=strides,
                                         pads_begin=pads_begin, pads_end=pads_end,
                                         dilations=dilations,
                                         mask=mask,
                                         auto_pad="EXPLICIT",
                                         deformable_group=deformable_groups, group=groups,
                                         bilinear_interpolation_pad=True,
                                         name='y')
    if bias is not None:
        s = graph.outputs()[0].get_partial_shape().get_shape()
        node_bias = ng.parameter(
            shape=bias.shape, name='bias', dtype=np.float32)
        target_shape = ng.constant(
            np.array([1, bias.shape[0], 1, 1], dtype=np.int32))
        new_bias = opset.reshape(node_bias, target_shape, special_zero=False)
        graph = opset.add(graph, new_bias, name='y')

    graph = ng.result(graph, name='y')

    parameters = [node_x, node_deform_values, node_w]
    inputs_dict = {'x': test_x, "deform_values": offset, "w": weight}
    if bias is not None:
        inputs_dict['bias'] = bias
        parameters.append(node_bias)

    function = ng.Function(
        graph, parameters, "deform_conv")

    ie_network = ng.function_to_cnn(function)
    ie = IECore()
    executable_network = ie.load_network(ie_network, 'CPU')
    output = executable_network.infer(inputs_dict)

    return output['y']


def deformable_conv(name: str, x, weight, offset, mask, bias, stride=1, padding=0, dilation=1, deformable_groups=1, groups=1):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_offset = pdpd.static.data(
            name='offset', shape=offset.shape, dtype=offset.dtype)
        node_weight = pdpd.static.data(
            name='weight', shape=weight.shape, dtype=weight.dtype)
        if mask is not None:
            node_mask = pdpd.static.data(
                name='mask', shape=mask.shape, dtype=mask.dtype)
        if bias is not None:
            node_bias = pdpd.static.data(
                name='bias', shape=bias.shape, dtype=bias.dtype)

        node_out = pdpd.vision.ops.deform_conv2d(node_x,
                                                 node_offset,
                                                 node_weight,
                                                 bias=node_bias if bias is not None else None,
                                                 stride=stride,  # int|list|tuple
                                                 padding=padding,  # int|list|tuple
                                                 dilation=dilation,  # int|list|tuple
                                                 deformable_groups=deformable_groups,  # int
                                                 groups=groups,  # int
                                                 mask=node_mask if mask is not None else None)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        feed_dict = {'x': x, 'offset': offset, 'weight': weight}
        inputs_list = [x, offset, weight]
        if mask is not None:
            feed_dict['mask'] = mask
            inputs_list.append(mask)
        if bias is not None:
            feed_dict['bias'] = bias
            inputs_list.append(bias)
        outs = exe.run(
            feed=feed_dict,
            fetch_list=node_out)

        # Save inputs in order of ngraph function, to facilite Fuzzy test,
        # which accepts inputs and outputs in this order as well.
        saveModel(name, exe, feedkeys=list(feed_dict.keys()), fetchlist=[node_out],
                  inputs=inputs_list, outputs=outs, target_dir=sys.argv[1] if len(sys.argv) > 1 else '.')

    # Run and compare with ngraph function. Disabled by default.
    if False:
        strides, paddings, dilations = pdpd_attr_to_list(
            stride, padding, dilation)

        ng_result = ngraph_deform_conv(
            x, weight, offset, mask, bias, strides, paddings, paddings, dilations, deformable_groups, groups)
        pdpd_result = outs

        match = np.all(np.isclose(
            pdpd_result, ng_result, rtol=1e-4, atol=1e-5))

        prefix_color = '\n\033[92m' if match else '\n\033[91m'
        print(prefix_color +
              'TestCase {} Result {} '.format(name, match) + '\033[0m\n')

        if not match:
            np.set_printoptions(precision=2)
            np.set_printoptions(suppress=True)

            print(prefix_color +
                  'pdpd_result: {}'.format(pdpd_result) + '\033[0m\n')
            print(prefix_color +
                  'ng_result: {}'.format(ng_result) + '\033[0m\n')

            # raise ValueError(name + ': OV result does not match PDPD!')

    return outs


def get_output_size(input_size, out_channels, kernel_size, stride, padding, dilation):
    # calculate output shape of conv2d
    def out_size(in_size, pad_size, dilation_size, kernel_size,
                 stride_size):
        return (in_size + 2 * pad_size -
                (dilation_size * (kernel_size - 1) + 1)) / stride_size + 1

    stride, padding, dilation = pdpd_attr_to_list(stride, padding, dilation)

    out_h = int(
        out_size(input_size[2], padding[0], dilation[0],
                 kernel_size[0], stride[0]))
    assert out_h > 0

    out_w = int(
        out_size(input_size[3], padding[1], dilation[1],
                 kernel_size[1], stride[1]))
    assert out_w > 0

    return [input_size[0], out_channels, out_h, out_w]


# ref: https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/vision/ops/deform_conv2d_en.html
# PaddlePaddle conv attributes padding, strides, dilation can optionally be int|lit|tuple,
# while they are list of int in ngraph.
def generator(input_size=[2, 8, 4, 4],  # NCHW
              out_channels=4,
              kernel_size=[3, 3],  # spatial_kernel
              padding=[0, 0],  # int|lit|tuple
              stride=[1, 1],  # int|lit|tuple
              dilation=[1, 1],  # int|lit|tuple
              groups=1, deformable_groups=1,  # int
              no_bias=True, no_mask=True, dtype='float32'):
    # np.random.seed(1)
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, ] * 2
    else:
        assert len(kernel_size) == 2

    assert np.mod(input_size[1], groups) == 0
    f_c = input_size[1] // groups
    filter_size = [out_channels, f_c] + kernel_size  # weight

    output_size = get_output_size(input_size, out_channels, kernel_size,
                                  stride, padding, dilation)  # output

    offset_c = 2 * deformable_groups * filter_size[
        2] * filter_size[3]
    mask_c = deformable_groups * filter_size[
        2] * filter_size[3]
    offset_size = [
        input_size[0], offset_c, output_size[2], output_size[3]  # offset
    ]
    mask_size = [
        input_size[0], mask_c, output_size[2], output_size[3]  # mask
    ]

    # data
    test_x = np.random.random(size=input_size).astype(dtype)

    weight = np.random.random(size=filter_size).astype(dtype)

    offset = (10 *
              np.random.uniform(-1, 1, size=offset_size)).astype(dtype)  # TODO: negative, fractioned

    mask = (10 *
            np.random.random(size=mask_size)).astype(dtype) if not no_mask else None

    bias = np.random.uniform(-1, 1,
                             size=(filter_size[0],)).astype(dtype) if not no_bias else None

    return [test_x, weight, offset, mask, bias]


def TEST1():
    data_x, data_weight, data_offset, data_mask, data_bias = generator()
    deformable_conv('deformable_conv_default', data_x,
                    data_weight, data_offset, data_mask, data_bias)


def TestWithPad():
    padding = 1
    data_x, data_weight, data_offset, data_mask, data_bias = generator(input_size=[1, 1, 4, 4],
                                                                       kernel_size=[3, 3], out_channels=1,
                                                                       padding=padding)
    deformable_conv('deformable_conv_with_pad', data_x,
                    data_weight, data_offset, data_mask, data_bias, padding=padding)


def TestWithPadTuple():
    padding = (3, 3)
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        padding=padding)
    deformable_conv('deformable_conv_with_pad_tuple', data_x,
                    data_weight, data_offset, data_mask, data_bias, padding=padding)


def TestWithPadList():
    padding = [3, 3]
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        padding=padding)
    deformable_conv('deformable_conv_with_pad_list', data_x,
                    data_weight, data_offset, data_mask, data_bias, padding=padding)


def TestWithStride():
    stride = 2
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        stride=stride)
    deformable_conv('deformable_conv_with_stride', data_x,
                    data_weight, data_offset, data_mask, data_bias, stride=stride)


def TestWithStrideTuple():
    stride = [2, 3]
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        stride=stride)
    deformable_conv('deformable_conv_with_stride_tuple', data_x,
                    data_weight, data_offset, data_mask, data_bias, stride=stride)


def TestWithStrideList():
    stride = [3, 2]
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        stride=stride)
    deformable_conv('deformable_conv_with_stride_list', data_x,
                    data_weight, data_offset, data_mask, data_bias, stride=stride)


def TestWithDilation():
    dilation = 2
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        input_size=[1, 1, 7, 7], dilation=dilation)
    deformable_conv('deformable_conv_with_dilation', data_x,
                    data_weight, data_offset, data_mask, data_bias, dilation=dilation)


def TestWithDilationTuple():
    dilation = (2, 3)
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        input_size=[1, 1, 7, 7], dilation=dilation)
    deformable_conv('deformable_conv_with_dilation_tuple', data_x,
                    data_weight, data_offset, data_mask, data_bias, dilation=dilation)


def TestWithDilationList():
    dilation = [3, 2]
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        input_size=[1, 1, 7, 7], dilation=dilation)
    deformable_conv('deformable_conv_with_dilation_list', data_x,
                    data_weight, data_offset, data_mask, data_bias, dilation=dilation)


def TestWithPadStrideDilation():
    dilation = [3, 2]
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        input_size=[1, 1, 7, 7], dilation=dilation, padding=1, stride=[3, 2])
    deformable_conv('deformable_conv_with_pad_stride_dilation', data_x,
                    data_weight, data_offset, data_mask, data_bias, dilation=dilation, padding=1, stride=[3, 2])


def TestWithGroup():
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        groups=2)
    deformable_conv('deformable_conv_with_groups', data_x,
                    data_weight, data_offset, data_mask, data_bias, groups=2)


def TestWithDeformableGroups():
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        deformable_groups=2)
    deformable_conv('deformable_conv_with_deformable_groups', data_x,
                    data_weight, data_offset, data_mask, data_bias, deformable_groups=2)


def TestWithGroupDeformableGroups():
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        groups=2, deformable_groups=2)
    deformable_conv('deformable_conv_with_groups_and_deformable_groups', data_x,
                    data_weight, data_offset, data_mask, data_bias, groups=2, deformable_groups=2)


def TestWithMask():
    data_x, data_weight, data_offset, data_mask, data_bias = generator(input_size=[2, 4, 3, 3], out_channels=1, kernel_size=[2, 2],
                                                                       no_mask=False)
    deformable_conv('deformable_conv_with_mask', data_x,
                    data_weight, data_offset, data_mask, data_bias)


def TestWithBias():
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        no_bias=False)
    deformable_conv('deformable_conv_with_bias', data_x,
                    data_weight, data_offset, data_mask, data_bias)


def TestWithMaskBias():
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
        no_mask=False, no_bias=False)
    deformable_conv('deformable_conv_with_mask_bias', data_x,
                    data_weight, data_offset, data_mask, data_bias)


def TestWithMaskBiasAllAttr():
    data_x, data_weight, data_offset, data_mask, data_bias = generator(input_size=[1, 4, 7, 7],
                                                                       dilation=[3, 2], padding=1, stride=[3, 2],
                                                                       no_mask=False, no_bias=False,
                                                                       groups=2, deformable_groups=2)
    deformable_conv('deformable_conv_full', data_x,
                    data_weight, data_offset, data_mask, data_bias,
                    groups=2, deformable_groups=2,
                    dilation=[3, 2], padding=1, stride=[3, 2])


if __name__ == "__main__":
    iter_loops = 1 # iterates default 1
    for i in range(iter_loops):
        TEST1()

        TestWithPad()
        TestWithPadTuple()
        TestWithPadList()

        TestWithStride()
        TestWithStrideTuple()
        TestWithStrideList()

        TestWithDilation()
        TestWithDilationTuple()
        TestWithDilationList()

        TestWithPadStrideDilation()

        TestWithGroup()
        TestWithDeformableGroups()

        TestWithGroupDeformableGroups()

        TestWithMask()
        TestWithBias()
        TestWithMaskBias()

        TestWithMaskBiasAllAttr()
