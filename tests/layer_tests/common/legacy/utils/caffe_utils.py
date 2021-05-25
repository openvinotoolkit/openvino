import copy
import os

import cv2
import numpy as np


os.environ['GLOG_minloglevel'] = '3'
import caffe



def preprocess_image(image_path, input_shape=(224, 224, 3)):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[0], input_shape[1]))
    img = img / 255
    img = img.astype(np.float32)
    return img


def summarize_graph(model_path, prototxt=None, reshape_net=None):
    caffe.set_mode_cpu()
    summary = dict(
        inputs=dict(),
        outputs=dict(),
        blob_layer_mapping=dict()  # mapping only for inputs and outputs (blob: layer, does not work for layers with multiple output blobs)
    )
    if prototxt is None:
        prototxt = os.path.splitext(model_path)[0] + '.prototxt'

    net = caffe.Net(prototxt, model_path, caffe.TEST)

    for input in net.inputs:
        if reshape_net and input in reshape_net.keys():
            net.blobs[input].reshape(*reshape_net[input])
        for key, value in net.top_names.items():
            if value[0] == input:
                summary['blob_layer_mapping'][input] = key
        summary['inputs'][input] = dict(shape=net.blobs[input].data.shape)
    outputs = list()


    for out in net.outputs:
        for layer in net._layer_names:
            if out in net.top_names[layer]:
                outputs.append(layer)
        #for key, value in net.top_names.items():
        #    if value == out:
        #        summary['blob_layer_mapping'][input] = key
    summary['outputs'] = outputs
    return summary


def collect_caffe_references(model_path, feed_dict, prototxt=None, reshape=True, mean_mode='no_mean', mean_file='',
                             mean_values='', squeeze_output=True):
    """
    Function to run caffe inference and collect results
    :param model_path: absolute path to .caffemodel
    :param feed_dict: dict with pairs of input blob names and input data.
    :return: return dict with of blob name and blob data.
    """
    _feed_dict = copy.deepcopy(feed_dict)
    caffe.set_mode_cpu()
    if prototxt is None:
        prototxt = os.path.splitext(model_path)[0] + '.prototxt'

    net = caffe.Net(prototxt, model_path, caffe.TEST)
    transformers = caffe.io.Transformer({_input: net.blobs[_input].data.shape for _input in net.inputs})
    if mean_mode != 'no_mean':
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open(mean_file, 'rb').read()
        blob.ParseFromString(data)
        mean_array = np.array(caffe.io.blobproto_to_array(blob))[0]
        if mean_mode == 'mean_values':
            mean = np.array(list(map(float, mean_values.strip('()').split(','))))
        elif mean_mode == 'mean_file':
            mean = mean_array[:, :transformers.inputs['data'][2], :transformers.inputs['data'][3]]
            mean = np.swapaxes(mean, 2, 0)
            mean = np.swapaxes(mean, 0, 1)

    for input in feed_dict:
        input_shape = _feed_dict[input].shape
        if mean_mode != 'no_mean':
            for i in range(len(_feed_dict[input])):
                _feed_dict[input][i] = _feed_dict[input][i] - mean

        if len(input_shape) == 4 and reshape:
            _feed_dict[input] = np.swapaxes(_feed_dict[input], 3, 1)
            _feed_dict[input] = np.swapaxes(_feed_dict[input], 2, 3)
            input_shape = _feed_dict[input].shape
            net.blobs[input].reshape(*input_shape)
        for i in range(len(_feed_dict[input])):
            net.blobs[input].data[i] = _feed_dict[input][i]

    net.forward()
    infer_res = dict()

    for out in net.outputs:
        for layer in net._layer_names:
            if out in net.top_names[layer]:
                if squeeze_output:
                    infer_res[layer] = np.squeeze(net.blobs[net.top_names[layer][0]].data)
                else:
                    infer_res[layer] = net.blobs[net.top_names[layer][0]].data

    return infer_res
