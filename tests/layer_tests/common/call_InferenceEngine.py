import cv2
import numpy as np
from common.layer_utils import IEInfer
from mo.utils.ir_engine.ir_engine import IREngine

from .constants import *


def score_model(model_path, image_path, device="CPU", batch=1, out_blob_name=None):
    """
    Run IE inference for checking accuracy
    :param model_path: path to IR for scoring
    :param image_path: path to image for scoring
    :param device: target device (default=CPU)
    :param batch: batch size (default=1)
    :param out_blob_name: name of output layer
    """

    data = cv2.imread(image_path)
    input_data = {}
    net = IREngine(model_path, model_path.replace(".xml", ".bin"))
    for layer_name in net.get_inputs():
        h, w = net.get_inputs()[layer_name][-2:]
        data = cv2.resize(data, (w, h))
        data = np.transpose(data, (2, 0, 1))
        data = np.expand_dims(data, axis=0)
        data = np.repeat(data, batch, axis=0)
        input_data[layer_name] = data

    ie_engine = IEInfer(device=device, model=model_path, weights=model_path.replace(".xml", ".bin"))
    result = ie_engine.infer(input_data=input_data)

    if out_blob_name and out_blob_name in result:
        return result[out_blob_name]
    else:
        raise RuntimeError("Scoring results haven't {} layer. Please use: {}".format(out_blob_name, result.keys()))


def compare_infer_results_with_caffe(infer_res, out_blob_name):
    try:
        from common.legacy.utils.common_utils import preprocess_input, allclose
        from common.legacy.utils.caffe_utils import summarize_graph, collect_caffe_references
    except ImportError:
        preprocess_input, summarize_graph, collect_caffe_references = None, None, None

    caffe_model = summarize_graph(os.path.join(caffe_models_path, 'network.caffemodel'))
    feed_dict = preprocess_input(input_paths=img_path, inputs=caffe_model['inputs'])

    caffe_ref = collect_caffe_references(model_path=os.path.join(caffe_models_path, 'network.caffemodel'),
                                         feed_dict=feed_dict)

    if out_blob_name in caffe_ref:
        caffe_ref = caffe_ref[out_blob_name]
    else:
        raise RuntimeError("Caffe results haven't {} layer ".format(out_blob_name))

    if allclose(np.squeeze(infer_res), caffe_ref, atol=caffe_eps, rtol=caffe_eps):
        return True
    else:
        print("Max diff is {}".format(np.array(abs(np.squeeze(infer_res) - caffe_ref)).max()))
        return False


def compare_infer_results_with_mxnet(infer_res, name, out_blob_name, input_shape):
    try:
        from common.legacy.utils.common_utils import preprocess_input, allclose
        from common.legacy.utils.mxnet_utils import summarize_graph, collect_mxnet_references
    except ImportError:
        preprocess_input, summarize_graph, collect_mxnet_references = None, None, None

    mxnet_model = summarize_graph(os.path.join(mxnet_models_path, name + '-0000.params'),
                                  {'data': {'shape': input_shape}})
    feed_dict = preprocess_input(input_paths=img_path, inputs=mxnet_model['inputs'])

    mxnet_ref = collect_mxnet_references(os.path.join(mxnet_models_path, name + '-0000.params'),
                                         feed_dict=feed_dict)

    if out_blob_name in mxnet_ref:
        mxnet_ref = mxnet_ref[out_blob_name]
    else:
        raise RuntimeError("MxNet results haven't {} layer ".format(out_blob_name))

    if allclose(np.squeeze(infer_res), mxnet_ref, atol=mxnet_eps, rtol=mxnet_eps):
        return True
    else:
        print("Max diff is {}".format(np.array(abs(np.squeeze(infer_res) - mxnet_ref)).max()))
        return False
