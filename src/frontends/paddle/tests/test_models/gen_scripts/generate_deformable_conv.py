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
        padding = list(p)
    elif isinstance(p, list):
        padding = p
    else:
        raise ValueError('unknown type of padding!')
    assert len(padding) == 2, 'len(padding) must be 2!'

    return strides, padding, dilations


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

        pdpd.static.nn.deform_conv2d

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

        # Save inputs in order of OpenVINO model, to facilitate Fuzzy test,
        # which accepts inputs and outputs in this order as well.
        feed_list = [node_x, node_offset, node_weight]
        if mask is not None:
            feed_list.append(node_mask)
        if bias is not None:
            feed_list.append(node_bias)
        saveModel(name, exe, feed_vars=feed_list, fetchlist=[node_out],
                  inputs=inputs_list, outputs=outs, target_dir=sys.argv[1] if len(sys.argv) > 1 else '.')


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
# while they are list of int in OpenVINO.
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

    offset = 10 * \
        np.random.uniform(size=offset_size).astype(
            'int').astype(dtype)  # TODO: negative, fractioned

    mask = 10 * \
        np.random.random(size=mask_size).astype(dtype) if not no_mask else None

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

    # data_mask = None
    # data_bias = None

    # data_x = np.random.random((1, 1, 3, 5)).astype('float32')
    # data_weight = np.random.random((1, 1, 3, 3)).astype('float32')
    # data_offset = np.random.random((1, 18, 1, 1)).astype('float32')

    # deformable_conv('deformable_conv_with_pad', data_x,
    #                 data_weight, data_offset, data_mask, data_bias, dilation=[1,2])


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


def TestWithMask():
    data_x, data_weight, data_offset, data_mask, data_bias = generator(
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


if __name__ == "__main__":
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

    TestWithGroup()
    TestWithDeformableGroups()

    TestWithMask()
    TestWithBias()
    TestWithMaskBias()
