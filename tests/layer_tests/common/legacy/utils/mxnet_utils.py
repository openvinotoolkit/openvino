from collections import namedtuple
import os
import mxnet as mx
import cv2
import numpy as np

import logging as log

logger = log.getLogger('mxnet')
logger.setLevel(log.CRITICAL)


def preprocess_image(image_path, color_format="BGR", input_shape=(3, 224, 224)):
    if color_format == "BGR":
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(image_path)

    img = cv2.resize(img, (input_shape[2], input_shape[3]))
    # convert into format (batch, RGB, width, height)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    img = img.astype(np.float32)
    return img


def summarize_graph(model_path, inputs):
    # take full path + name without "-NNNN" part
    model_path = os.path.splitext(model_path)[0][:-5]
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)

    summary = dict(
        inputs=dict(),
        outputs=dict()
    )
    data_shapes = list()
    for input in inputs:
        data_shapes.append((input, inputs[input]['shape']))
    mod.bind(for_training=False, data_shapes=data_shapes)
    for input in mod.data_shapes:
        summary['inputs'][input.name] = dict(shape=input.shape)

    output_names = []
    for names in mod.output_names:
        output_names.append(names.replace("_output", ""))

    summary['outputs'] = output_names
    return summary


def collect_mxnet_references(model_path, feed_dict, mean_mode='no_mean'):
    """
    return output blob of mxnet infer results 
    :param model_path: path to mxnet model
    :param feed_dict: dictionary with data
    key = layer name; value = data
    :return: dictionary: key - output layer name, value - numpy ndarray
    """

    if len(feed_dict.keys()) > 1:
        raise RuntimeError("MXNet multi input inference not implemented yet!")

    # take full path + name without "-NNNN" part
    model_path = os.path.splitext(model_path)[0][:-5]
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)

    mx_data = []
    for inputl, data in feed_dict.items():
        _data = data
        if len(data.shape) == 4:
            # Reshape only data with images in NHWC format
            _data = np.transpose(_data, axes=(0, 3, 1, 2))
        mx_data.append(mx.nd.array(_data))
    batch = mx_data[0].shape[0]

    data_iter = mx.io.NDArrayIter(data=mx_data, label=None, batch_size=batch)
    data_batch = mx.io.DataBatch(data=mx_data)

    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=data_iter.provide_data)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    mod.forward(data_batch)

    infer_res = dict()
    for outputl, mx_out in zip(mod.output_names, mod.get_outputs()):
        infer_res[outputl.replace("_output", "")] = np.squeeze(mx_out.asnumpy())

    return infer_res
