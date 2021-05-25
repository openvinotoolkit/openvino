import numpy as np

from common.legacy.generic_ir_comparator.layers import Layer


def get(l: list, idx: int, default=None):
    try:
        return l[idx]
    except IndexError:
        return default


# --------------------CALCULATION SHAPES--------------------
def calc_out_shape_input_layer(_layer: Layer, shape=None):
    """
    :param _layer:
    :param shape: not 1*3*224 if need
    :return:
    """
    if not shape:
        return 1, 3, 224, 224
    else:
        return shape


def calc_out_shape_correlation(layer: Layer):
    input_dims = layer.get_inputs_shape(layer.get_inputs_names()[0])

    outn = np.zeros(4, dtype=int)
    outn[0] = input_dims[0]

    paddedbottomheight = input_dims[2]
    paddedbottomwidth = input_dims[3] + 2 * layer.attrs['pad']

    kernel_radius_ = (layer.attrs['kernel_size'] - 1) / 2
    border_size_ = layer.attrs['max_displacement'] + kernel_radius_

    outn[3] = np.ceil((float)(paddedbottomwidth - border_size_ * 2) / layer.attrs['stride_1'])
    outn[2] = np.ceil((float)(paddedbottomheight - kernel_radius_ * 2) / layer.attrs['stride_1'])

    neighborhood_grid_radius_ = layer.attrs['max_displacement'] / layer.attrs['stride_2']

    if layer.attrs['single_direction'] != 0:
        neighborhood_grid_width_ = neighborhood_grid_radius_ + 1
    else:
        neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1

    outn[1] = neighborhood_grid_width_ * neighborhood_grid_width_
    return outn


def calc_out_shape_theta_layer(layer: Layer):
    return calc_out_shape_input_layer(layer, (1, 6))


def calc_same_out_shape(layer: Layer):
    return layer.get_inputs_shape(layer.get_inputs_names()[0])


def calc_mult_out_shape(layer: Layer):
    multiplier = 2  # Let's multiply input shape on 2
    return np.multiply(layer.get_inputs_shape(layer.get_inputs_names()[0]), multiplier)


def caffe_calc_out_accum_layer(layer: Layer):
    num = layer.get_inputs_shape(layer.get_inputs_names()[0])[0]
    channels = 0
    if 'have_reference' in layer.attrs:
        for i in layer.get_inputs_names():
            channels += layer.get_inputs_shape(i)[1]
        height = layer.get_inputs_shape(layer.get_inputs_names()[-1])[2]
        width = layer.get_inputs_shape(layer.get_inputs_names()[-1])[3]
    else:
        maxheight = -1
        maxwidth = -1
        for i in layer.get_inputs_names():
            channels += layer.get_inputs_shape(i)[1]
            maxheight = layer.get_inputs_shape(i)[2] if layer.get_inputs_shape(i)[2] > maxheight else maxheight
            maxwidth = layer.get_inputs_shape(i)[3] if layer.get_inputs_shape(i)[3] > maxwidth else maxwidth

        if 'size_divisible_by' in layer.attrs:
            height = int(np.ceil(maxheight / layer.attrs['size_divisible_by']) * layer.attrs['size_divisible_by'])
            width = int(np.ceil(maxwidth / layer.attrs['size_divisible_by']) * layer.attrs['size_divisible_by'])
        else:
            height = layer.attrs['top_height']
            width = layer.attrs['top_width']

        if height < maxheight or width < maxheight:
            height = maxheight
            width = maxheight

    return num, channels, height, width


def caffe_calc_out_arg_max_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    return tuple([layer.attrs['top_k'] if i == layer.attrs['axis'] else val for i, val in enumerate(input_dims)])


def caffe_calc_out_shape_concat_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    concated_shape = 0
    for k in layer.inputs.keys():
        for i in layer.inputs[k]:
            concated_shape += i[1][int(layer.attrs['axis'])]
    out_shape_concat = list(input_dims)
    out_shape_concat[int(layer.attrs['axis'])] = concated_shape
    return tuple(out_shape_concat)


def caffe_calc_out_shape_conv_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    [kernel, pad, stride] = get_conv_parameters(layer)
    h_0 = int((input_dims[2] + 2 * pad[0] - kernel[0]) / stride[0] + 1)
    w_0 = int((input_dims[3] + 2 * pad[1] - kernel[1]) / stride[1] + 1)
    return input_dims[0], layer.attrs['output'], h_0, w_0


def caffe_calc_out_shape_crop_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    offsets = layer.attrs['offset'].split(',')
    return tuple([x - int(get(offsets, i, 0)) for i, x in enumerate(input_dims)])


def caffe_calc_out_shape_ctc_decoder_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    if len(layer.get_inputs_names()) == 2:
        return input_dims[1], input_dims[0], 1, 1
    elif len(layer.get_inputs_names()) == 3:
        return input_dims[1], input_dims[0], 1, 1


def caffe_calc_out_shape_deconv_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    [kernel, pad, stride] = get_conv_parameters(layer)
    h_0 = int((input_dims[2] - 1) * stride[0] + kernel[0] - 2 * pad[0])
    w_0 = int((input_dims[3] - 1) * stride[1] + kernel[1] - 2 * pad[1])
    return input_dims[0], layer.attrs['output'], h_0, w_0


def caffe_calc_out_shape_detection_output_layer(layer: Layer):
    return 1, 1, layer.attrs['keep_top_k'], 7


def caffe_calc_out_shape_flatten_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    axis = int(layer.attrs['axis'])
    end_axis = int(layer.attrs['end_axis']) if 'end_axis' in layer.attrs else 3

    prod_axes = np.prod(input_dims[axis: end_axis + 1])
    return np.array([*input_dims[0: axis], prod_axes, *input_dims[end_axis + 1:]], dtype=np.int64)


def caffe_calc_out_shape_fullyconnected_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    return input_dims[0], layer.attrs['out_size']


def caffe_calc_out_shape_interp_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)

    height_in_eff = input_dims[2] + layer.attrs.get('pad_beg', 0) + layer.attrs.get('pad_end', 0)
    width_in_eff = input_dims[3] + layer.attrs.get('pad_beg', 0) + layer.attrs.get('pad_end', 0)

    height_out, width_out = 0, 0
    if not layer.attrs.get('shrink_factor') == 1 and layer.attrs.get('zoom_factor') == 1:
        height_out = int((height_in_eff - 1) / layer.attrs.get('shrink_factor')) + 1
        width_out = int((width_in_eff - 1) / layer.attrs.get('shrink_factor')) + 1
    elif layer.attrs.get('shrink_factor') == 1 and not layer.attrs.get('zoom_factor') == 1:
        height_out = height_in_eff * layer.attrs.get('zoom_factor')
        width_out = width_in_eff * layer.attrs.get('zoom_factor')
    elif not (layer.attrs.get('shrink_factor') == 1 or layer.attrs.get('zoom_factor') == 1):
        height_out = int((height_in_eff - 1) / layer.attrs.get('shrink_factor')) + 1
        width_out = int((width_in_eff - 1) / layer.attrs.get('shrink_factor')) + 1
        height_out = height_out + (height_out - 1) * (layer.attrs.get('zoom_factor') - 1)
        width_out = width_out + (width_out - 1) * (layer.attrs.get('zoom_factor') - 1)
    elif not (layer.attrs.get('height') == 0 or layer.attrs.get('width') == 0):
        height_out = layer.attrs.get('height')
        width_out = layer.attrs.get('width')

    return input_dims[0], input_dims[1], height_out, width_out


def caffe_calc_out_shape_permute_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    return tuple([input_dims[int(x)] for x in layer.attrs['order']])


def caffe_calc_out_shape_pool_layer(layer: Layer):
    # https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cpp
    input_dims = calc_same_out_shape(layer)
    [kernel, pad, stride] = get_conv_parameters(layer)
    pooled_height_ = int(np.ceil((input_dims[2] + 2 * pad[0] - kernel[0]) / stride[0])) + 1
    pooled_width_ = int(np.ceil((input_dims[3] + 2 * pad[1] - kernel[1]) / stride[1])) + 1
    if pad[0] or pad[1]:
        if (pooled_height_ - 1) * stride[0] >= input_dims[2] + pad[0]:
            pooled_height_ -= 1
        if (pooled_width_ - 1) * stride[1] >= input_dims[3] + pad[1]:
            pooled_width_ -= 1
    return input_dims[0], input_dims[1], pooled_height_, pooled_width_


def caffe_calc_out_shape_prior_box_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    flip = int(layer.attrs['flip'])
    aspect_ratio_len = len(layer.attrs['aspect_ratio'].split(','))
    min_size_len = len(layer.attrs['min_size']) if isinstance(layer.attrs['min_size'], list) else 1
    max_size_len = len(layer.attrs['max_size']) if isinstance(layer.attrs['max_size'], list) else 1
    num_ratios = ((flip + 1) * aspect_ratio_len + 1) * min_size_len + max_size_len

    return 1, 2, input_dims[2] * input_dims[3] * num_ratios * 4


def caffe_calc_out_shape_prior_box_clustered_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    num_priors = 1  # len(layer.attrs['width'].split(',')) TODO: add possible to use multiple widths

    return 1, 2, input_dims[2] * input_dims[3] * num_priors * 4


def caffe_calc_out_shape_ps_roi_pool_layer(layer: Layer):
    num = layer.get_inputs_shape(layer.get_inputs_names()[1])[0]
    channels = layer.get_inputs_shape(layer.get_inputs_names()[0])[1]
    return num, channels, layer.attrs['group_size'], layer.attrs['group_size']


def caffe_calc_out_shape_region_yolo_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    return input_dims[0], np.prod(input_dims[1:-1])


def caffe_calc_out_shape_resample_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    # TODO: calculate top_width and top_height if height and width not given.
    return input_dims[0], input_dims[1], int(layer.attrs['height']), int(layer.attrs['width'])


def caffe_calc_out_shape_reshape_layer(layer: Layer):
    return tuple(map(int, layer.attrs['dim'].split(',')))


def caffe_calc_out_shape_roi_pool_layer(layer: Layer):
    num = layer.get_inputs_shape(layer.get_inputs_names()[1])[0]
    channels = layer.get_inputs_shape(layer.get_inputs_names()[0])[1]
    return num, channels, layer.attrs['pooled_h'], layer.attrs['pooled_w']


def caffe_calc_out_shape_simpler_nms_layer(layer: Layer):
    return layer.attrs['post_nms_topn'], 5


def caffe_calc_out_shape_slice_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    slice_points = sorted(layer.attrs['slice_point'])
    operated_point = 0
    sliced_shapes = []
    for point in slice_points:
        sliced_shapes.append(point - operated_point)
        operated_point = point
    sliced_shapes.append(input_dims[layer.attrs['axis']] - operated_point)

    out_shapes = []
    for s in sliced_shapes:
        out_shapes.append(tuple([s if i == int(layer.attrs['axis']) else val for i, val in enumerate(input_dims)]))
    return out_shapes


def caffe_calc_out_shape_st_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    return input_dims[0], input_dims[1], layer.attrs['output_H'], layer.attrs['output_W']


def caffe_calc_out_shape_tile_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    return tuple([val * layer.attrs['tiles'] if i == layer.attrs['axis'] else val for i, val in enumerate(input_dims)])


def tf_calc_out_shape_concat_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    concat_shape = 0
    for k in layer.inputs.keys():
        for i in layer.inputs[k]:
            concat_shape += i[1][int(layer.attrs['axis'])]
    return tuple([concat_shape if i == int(layer.attrs['axis']) else val for i, val in enumerate(input_dims)])


def caffe_calc_data_augmentation_layer(layer: Layer):
    out_dims = calc_same_out_shape(layer)
    h = layer.attrs['crop_height'] if layer.attrs['crop_height'] != 0 else out_dims[2]
    w = layer.attrs['crop_width'] if layer.attrs['crop_width'] != 0 else out_dims[3]
    return out_dims[0], out_dims[1], h, w


def tf_calc_out_shape_conv_layer_pad_same_upper(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    stride = layer.attrs['strides']
    h_0 = np.int64(np.ceil(float(input_dims[2]) / float(stride[0])))
    w_0 = np.int64(np.ceil(float(input_dims[3]) / float(stride[1])))
    return input_dims[0], layer.attrs['output'], h_0, w_0


def tf_calc_out_shape_conv_layer_pad_valid(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    [kernel, pad, stride] = get_conv_parameters(layer)
    h_0 = np.int64(np.math.ceil((input_dims[2] + 2 * pad[0] - kernel[0] + 1) / (stride[0])))
    w_0 = np.int64(np.math.ceil((input_dims[3] + 2 * pad[1] - kernel[1] + 1) / (stride[1])))
    return input_dims[0], layer.attrs['output'], h_0, w_0


def tf_calc_out_shape_deconv_layer_pad_same_upper(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    # kernel = layer.attrs['kernel']
    stride = layer.attrs['strides']
    h_0 = int(np.ceil(input_dims[2] * stride[0]))
    w_0 = int(np.ceil(input_dims[3] * stride[1]))
    # if h_0 % stride[0] == 0:
    #     pad_along_height = max(kernel[0] - stride[0], 0)
    # else:
    #     pad_along_height = max(kernel[0] - (h_0 % stride[0]), 0)
    # if w_0 % stride[1] == 0:
    #     pad_along_width = max(kernel[1] - stride[1], 0)
    # else:
    #     pad_along_width = max(kernel[1] - (w_0 % stride[1]), 0)
    # layer.attrs['pad_x'] = int(pad_along_width / 2)
    # layer.attrs['pad_y'] = int(pad_along_height / 2)
    return input_dims[0], layer.attrs['output'], h_0, w_0


def tf_calc_out_shape_flatten_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    axis = int(layer.attrs['axis'])
    end_axis = int(layer.attrs['end_axis'])

    out_shape = []
    do_prod = True
    for i in range(0, len(input_dims)):
        if i < axis or i > end_axis:
            out_shape.append(input_dims[i])
        elif do_prod:
            out_shape.append(np.prod(input_dims[axis:end_axis + 1]))
            do_prod = False
    return tuple(out_shape)


def tf_calc_out_shape_fullyconnected_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    return input_dims[0], layer.attrs['out_size']


def tf_calc_out_shape_pool_layer_pad_same(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    kernel = layer.attrs['kernel']
    stride = layer.attrs['strides']
    h_0 = np.math.ceil(float(input_dims[2]) / float(stride[0]))
    w_0 = np.math.ceil(float(input_dims[3]) / float(stride[1]))
    return input_dims[0], input_dims[1], h_0, w_0


def tf_calc_out_shape_pool_layer_pad_valid(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    [kernel, pad, stride] = get_conv_parameters(layer)
    h_0 = np.int64(np.math.ceil((input_dims[2] + 2 * pad[1] - kernel[1] + 1) / (stride[1])))
    w_0 = np.int64(np.math.ceil((input_dims[3] + 2 * pad[0] - kernel[0] + 1) / (stride[0])))
    return input_dims[0], input_dims[1], h_0, w_0


def tf_calc_out_shape_reshape_layer(layer: Layer):
    return tuple(map(int, layer.attrs['dim'].split(',')))


def tf_calc_out_shape_slice_layer(layer: Layer):
    return int(layer.attrs['size']),


def tf_calc_out_shape_split_layer(layer: Layer, num_split=None):
    input_dims = calc_same_out_shape(layer)
    num_or_size_split = num_split if num_split else layer.attrs['num_split']

    out_shapes = []
    if isinstance(num_or_size_split, list):
        for s in num_or_size_split:
            out_shapes.append(tuple(
                [s if i == int(layer.attrs['axis']) else val for i, val in enumerate(input_dims)]))
    else:
        for _ in range(0, num_or_size_split):
            out_shapes.append(tuple(
                [val / num_or_size_split if i == int(layer.attrs['axis']) else val for i, val in
                 enumerate(input_dims)]))
    return out_shapes


def tf_calc_out_shape_splitv_layer(layer: Layer):
    return tf_calc_out_shape_split_layer(layer, 2)


def tf_calc_out_shape_tile_layer(layer: Layer):
    input_dims = calc_same_out_shape(layer)
    return tuple([val * layer.attrs['tiles'] if i == layer.attrs['axis'] else val for i, val in enumerate(input_dims)])


def mxnet_calc_out_shape_conv_layer(layer: Layer):
    data_shape = calc_same_out_shape(layer)
    [kernel, pad, stride] = get_conv_parameters(layer)
    dilation = layer.attrs.get('dilations')
    h_0 = calculate_single_dim(data_shape[2], pad[0], kernel[0], stride[0], dilation[0])
    w_0 = calculate_single_dim(data_shape[3], pad[1], kernel[1], stride[1], dilation[1])
    return data_shape[0], layer.attrs['output'], h_0, w_0


def mxnet_calc_out_shape_deconv_layer(layer: Layer):
    data_shape = calc_same_out_shape(layer)
    [kernel, pad, stride] = get_conv_parameters(layer)
    h_0 = int((data_shape[2] - 1) * stride[0] + kernel[0] - 2 * pad[0])
    w_0 = int((data_shape[3] - 1) * stride[1] + kernel[1] - 2 * pad[1])
    return data_shape[0], layer.attrs['output'], h_0, w_0


def mxnet_calc_out_shape_multi_box_prior_layer(layer: Layer):
    data_shape = calc_same_out_shape(layer)
    num_ratios = len(layer.attrs.get('aspect_ratio'))
    res_prod = data_shape[2] * data_shape[3] * (2 + num_ratios * 2) * 4
    return np.array([1, 2, res_prod // 2], dtype=np.int64)


def mxnet_calc_out_shape_flatten_layer(layer: Layer):
    input_shape = calc_same_out_shape(layer)
    axis = get_canonical_axis_index(input_shape, layer.attrs["axis"])
    end_axis = layer.attrs["end_axis"] if layer.attrs.get("end_axis") else -1
    end_axis = get_canonical_axis_index(input_shape, end_axis)
    prod_axes = np.prod(input_shape[axis: end_axis + 1])
    return np.array([*input_shape[0: axis], prod_axes, *input_shape[end_axis + 1:]], dtype=np.int64)


def mxnet_calc_out_shape_reshape_layer(layer: Layer):
    return layer.attrs['dim']


def mxnet_calc_out_shape_up_sampling_layer(layer: Layer):
    input_shape = layer.get_inputs_shape(layer.get_inputs_names()[0])
    batch = input_shape[0]
    channel = input_shape[1]
    y = input_shape[2] * layer.attrs["factor"]
    x = input_shape[3] * layer.attrs["factor"]
    return batch, channel, y, x


def mxnet_calc_out_shape_pooling_layer(layer: Layer):
    input_dims = layer.get_inputs_shape(layer.get_inputs_names()[0])
    [kernel, pad, stride] = get_conv_parameters(layer)
    # FIXME: ADD global_pool usage when global_pool==True
    global_pool = layer.attrs["global_pool"]
    # FIXME: Rewrite like in MO - inference-engine/model-optimizer-tensorflow/blob/master/mo/front/common/partial_infer/pooling.py
    pool_convention = layer.attrs["convention"]
    rounding = np.floor
    if pool_convention == "full":
        rounding = np.ceil
    pooled_height_ = int(rounding((input_dims[2] + 2 * pad[0] - kernel[0]) / stride[0])) + 1
    pooled_width_ = int(rounding((input_dims[3] + 2 * pad[1] - kernel[1]) / stride[1])) + 1
    if pad[0] or pad[1]:
        if (pooled_height_ - 1) * stride[0] >= input_dims[2] + pad[0]:
            pooled_height_ -= 1
        if (pooled_width_ - 1) * stride[1] >= input_dims[3] + pad[1]:
            pooled_width_ -= 1
    return input_dims[0], input_dims[1], pooled_height_, pooled_width_


# ------------------END CALCULATION SHAPES------------------


def get_conv_parameters(layer: Layer):
    assert layer.attrs.get('pads_begin') == layer.attrs.get('pads_end'), \
        "Attributes 'pads_begin' and 'pads_end' are different. Please, update tests to support this case"
    pad = layer.attrs.get('pads_begin')
    return layer.attrs.get('kernel'), pad, layer.attrs.get('strides')


def calculate_single_dim(input_val, padding, kernel, stride, dilate):
    return (input_val + 2 * padding - ((kernel - 1) * dilate + 1)) // stride + 1


def get_canonical_axis_index(shape, axis):
    return len(shape) + axis if axis < 0 else axis
