import mxnet as mx
import numpy as np


def input_to_symbol(layer, _input=None):
    if layer.attrs.get("layer_name") is None:
        name = layer.name
    else:
        name = layer.attrs.get("layer_name")

    weights_shape = layer.outputs[layer.name][0][1]
    weights = mx.ndarray.random.normal(0, 255, weights_shape)

    if hasattr(layer, "params"):
        layer.params = {"arg:data": weights}
    else:
        setattr(layer, "params", {"arg:data": weights})
    return mx.symbol.Variable(name)


def weights_to_symbol(layer, _input=None):
    weights_shape = layer.outputs[layer.name][0][1]
    weights = mx.ndarray.ones(weights_shape)
    layer.weights = np.reshape(weights.asnumpy(), (np.prod(weights_shape),))

    if hasattr(layer, "params"):
        layer.params = {"arg:{}".format(layer.name): weights}
    else:
        setattr(layer, "params", {"arg:{}".format(layer.name): weights})
    return mx.symbol.Variable(layer.name, shape=weights_shape)


def activation_to_symbol(layer, _input=None):
    act_type = layer.attrs["act_type"]
    return mx.symbol.Activation(data=_input[0].framework_representation_def, name=layer.name, act_type=act_type)


def broadcast_mul_to_symbol(layer, _input=None):
    # delete Weights with shape from layer.inputs
    layer.inputs.pop(_input[0].name)
    return mx.symbol.broadcast_mul(lhs=_input[1].framework_representation_def,
                                   rhs=_input[0].framework_representation_def, name=layer.name)


def relu_to_symbol(layer, _input=None):
    return mx.symbol.relu(data=_input[0].framework_representation_def, name=layer.name)


def sigmoid_to_symbol(layer, _input=None):
    return mx.symbol.sigmoid(data=_input[0].framework_representation_def, name=layer.name)


def eltwise_to_symbol(layer, _input=None):
    operation = layer.attrs["operation"]
    if operation == "sum":
        return eltwise_sum_to_symbol(layer, _input)
    elif operation == "mul":
        return eltwise_mul_to_symbol(layer, _input)


def eltwise_sum_to_symbol(layer, _input=None):
    return mx.symbol.elemwise_add(lhs=_input[0].framework_representation_def,
                                  rhs=_input[1].framework_representation_def, name=layer.name)


def eltwise_mul_to_symbol(layer, _input=None):
    return mx.symbol.elemwise_mul(lhs=_input[0].framework_representation_def,
                                  rhs=_input[1].framework_representation_def, name=layer.name)


def l2normalization_to_symbol(layer, _input=None):
    eps = layer.attrs["eps"]
    mode = layer.attrs["mode"]
    return mx.symbol.L2Normalization(data=_input[0].framework_representation_def, eps=eps, mode=mode)


def lrn_to_symbol(layer, _input=None):
    alpha = layer.attrs["alpha"]
    beta = layer.attrs["beta"]
    knorm = layer.attrs["knorm"]
    nsize = layer.attrs["local_size"]
    return mx.symbol.LRN(data=_input[0].framework_representation_def, alpha=alpha, beta=beta, knorm=knorm,
                         nsize=nsize, name=layer.name)


def plus_to_symbol(layer, _input=None):
    return _input[0].framework_representation_def + _input[1].framework_representation_def


def copy_to_symbol(layer, _input=None):
    import copy
    return copy.deepcopy(_input[0].framework_representation_def)


def concat_to_symbol(layer, _input=None):
    dim = layer.attrs["axis"]
    return mx.symbol.concat(_input[0].framework_representation_def, _input[1].framework_representation_def,
                            dim=dim, name=layer.name)


def reshape_to_symbol(layer, _input=None):
    shape = layer.attrs["dim"]
    return mx.symbol.reshape(data=_input[0].framework_representation_def, shape=shape, name=layer.name)


def up_sampling_to_symbol(layer, _input=None):
    scale = int(layer.attrs["factor"])
    sample_type = "nearest"
    return mx.symbol.UpSampling(_input[0].framework_representation_def, sample_type=sample_type, scale=scale, name=layer.name)


def dropout_to_symbol(layer, _input=None):
    return mx.symbol.Dropout(data=_input[0].framework_representation_def, name=layer.name)


def fullyconected_to_symbol(layer, _input=None):
    num_hidden = layer.attrs["out_size"]

    input_shape = layer.get_inputs_shape(layer.get_inputs_names()[0])
    weights_shape = (num_hidden, np.prod(input_shape))
    weights = mx.ndarray.random.normal(-1, 1, weights_shape)
    layer.weights = np.reshape(weights.asnumpy(), (np.prod(weights_shape),))
    weights_dict = {"arg:{}_weight".format(layer.name): weights}

    if hasattr(layer, "params"):
        layer.params = weights_dict
    else:
        setattr(layer, "params", weights_dict)

    no_bias = layer.attrs["no_bias"]
    if not no_bias:
        bias_shape = (num_hidden, )
        bias = mx.ndarray.random.normal(-1, 1, bias_shape)
        layer.biases = np.reshape(bias.asnumpy(), (np.prod(bias_shape),))
        layer.params.update({"arg:{}_bias".format(layer.name): bias})

    return mx.symbol.FullyConnected(data=_input[0].framework_representation_def, num_hidden=num_hidden,
                                    no_bias=no_bias, name=layer.name)


def conv_to_symbol(layer, _input=None):
    dilation = layer.attrs.get("dilations")
    kernel = layer.attrs["kernel"]
    stride = layer.attrs["strides"]
    pad = layer.attrs["pads_begin"]
    num_filter = layer.attrs["output"]

    weights_shape = (num_filter, 3, kernel[0], kernel[1])
    weights = mx.ndarray.random.normal(-1, 1, weights_shape)
    layer.weights = np.reshape(weights.asnumpy(), (np.prod(weights_shape), ))
    weights_dict = {"arg:{}_weight".format(layer.name): weights}

    if hasattr(layer, "params"):
        layer.params = weights_dict
    else:
        setattr(layer, "params", weights_dict)

    no_bias = layer.attrs["no_bias"]
    if not no_bias:
        bias_shape = (num_filter, )
        bias = mx.ndarray.random.normal(-1, 1, bias_shape)
        layer.biases = np.reshape(bias.asnumpy(), (np.prod(bias_shape),))
        layer.params.update({"arg:{}_bias".format(layer.name): bias})

    return mx.symbol.Convolution(data=_input[0].framework_representation_def, name=layer.name, kernel=kernel,
                                 stride=stride, pad=pad, dilate=dilation, num_filter=num_filter, no_bias=no_bias)


def deconv_to_symbol(layer, _input=None):
    dilation = layer.attrs.get("dilations")
    kernel = layer.attrs["kernel"]
    stride = layer.attrs["strides"]
    pad = layer.attrs["pads_begin"]
    num_filter = layer.attrs["output"]

    weights_shape = (3, num_filter, kernel[0], kernel[1])
    weights = mx.ndarray.random.normal(-1, 1, weights_shape)
    layer.weights = np.reshape(weights.asnumpy(), (np.prod(weights_shape),))
    weights_dict = {"arg:{}_weight".format(layer.name): weights}

    if hasattr(layer, "params"):
        layer.params = weights_dict
    else:
        setattr(layer, "params", weights_dict)

    no_bias = layer.attrs["no_bias"]
    if not no_bias:
        bias_shape = (num_filter, )
        bias = mx.ndarray.random.normal(-1, 1, bias_shape)
        layer.biases = np.reshape(bias.asnumpy(), (np.prod(bias_shape),))
        layer.params.update({"arg:{}_bias".format(layer.name): bias})

    return mx.symbol.Deconvolution(data=_input[0].framework_representation_def, name=layer.name, kernel=kernel,
                                   stride=stride, pad=pad, dilate=dilation, num_filter=num_filter, no_bias=no_bias)


def soft_max_output_to_symbol(layer, _input=None):
    if layer.attrs.get("layer_name") is None:
        name = layer.name
    else:
        name = layer.attrs.get("layer_name")
    return mx.symbol.SoftmaxOutput(data=_input[0].framework_representation_def, name=name)


def soft_max_activation_to_symbol(layer, _input=None):
    if layer.attrs.get("layer_name") is None:
        name = layer.name
    else:
        name = layer.attrs.get("layer_name")
    mode = 'channel' if layer.attrs["axis"] == 2 else 'instance'
    return mx.symbol.SoftmaxActivation(data=_input[0].framework_representation_def, mode=mode, name=name)


def flatten_to_symbol(layer, _input=None):
    return mx.symbol.flatten(data=_input[0].framework_representation_def, name=layer.name)


def transpose_to_symbol(layer, _input=None):
    axes = layer.attrs["order"]
    return mx.symbol.transpose(data=_input[0].framework_representation_def, axes=axes, name=layer.name)


def pool_to_symbol(layer, _input=None):
    kernel = layer.attrs["kernel"]
    stride = layer.attrs["strides"]
    pad = layer.attrs["pads_begin"]

    pool_type = layer.attrs["pool_method"]
    convention = layer.attrs["convention"]
    global_pool = layer.attrs["global_pool"]
    return mx.symbol.Pooling(data=_input[0].framework_representation_def, global_pool=global_pool,
                             kernel=kernel,
                             stride=stride, pad=pad, pool_type=pool_type,
                             pooling_convention=convention, name=layer.name)


def multi_box_prior_to_symbol(layer, _input=None):
    sizes = [i for i in layer.attrs.get('min_size')] if layer.attrs.get("min_size") is not None else [1]
    ratios = layer.attrs.get('aspect_ratio') if layer.attrs.get('aspect_ratio') is not None else [1]
    clip = bool(layer.attrs.get('clip')) if layer.attrs.get('clip') is not None else 0

    steps = (layer.attrs.get('step'),
             layer.attrs.get('step')) if layer.attrs.get('step') is not None else [-1, -1]

    offsets = (layer.attrs.get('offset'),
               layer.attrs.get('offset')) if layer.attrs.get('offset') is not None else [0.5, 0.5]

    # Rewrite attrs like in inference-engine/model-optimizer-tensorflow/blob/
    # master/mo/front/common/partial_infer/multi_box_prior.py
    # get_input_shapes as calc_same_out_shape from infer_shapes.py
    data_shape = layer.get_inputs_shape(layer.get_inputs_names()[0])
    img_shape = layer.get_inputs_shape(layer.get_inputs_names()[1])

    layer.attrs['aspect_ratio'] = ",".join(str(float(i)) for i in layer.attrs.get('aspect_ratio'))
    layer.attrs['min_size'] = [float(i * data_shape[2]) for i in layer.attrs['min_size']]
    layer.attrs['max_size'] = ''
    if layer.attrs['step'] != -1:
        layer.attrs['step'] = float(img_shape[2] * layer.attrs['step'])
    else:
        layer.attrs['step'] = float(img_shape[2] / data_shape[2])

    return mx.contrib.symbol.MultiBoxPrior(data=_input[0].framework_representation_def, name=layer.name,
                                           sizes=sizes, ratios=ratios, clip=clip, steps=steps, offsets=offsets)
