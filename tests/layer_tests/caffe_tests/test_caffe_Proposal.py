import itertools
import logging as lg

import pytest
from caffe_tests.conftest import generate_tests
from common.caffe_layers_representation import *
from common.call_InferenceEngine import score_model, compare_infer_results_with_caffe
from common.call_ModelOptimizer import generate_ir_from_caffe
from common.constants import *
from common.infer_shapes import *
from common.legacy.generic_ir_comparator import *


def get_proposal_params(ie_device=None, precision=None, feat_stride=None, base_size=None, min_size=None, ratio=None,
                        scale=None,
                        pre_nms_topn=None, post_nms_topn=None, nms_thresh=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param scale:
    :param ratio:
    :param min_size:
    :param base_size:
    :param nms_thresh:
    :param pre_nms_topn:
    :param post_nms_topn:
    :param feat_stride:
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if feat_stride:
        feat_stride_params = feat_stride
    else:
        feat_stride_params = [16]

    if base_size:
        base_size_params = base_size
    else:
        base_size_params = [16]

    if min_size:
        min_size_params = min_size
    else:
        min_size_params = [16]

    if ratio:
        ratio_params = ratio
    else:
        ratio_params = [1]

    if scale:
        scale_params = scale
    else:
        scale_params = [1]

    if pre_nms_topn:
        pre_nms_topn_params = pre_nms_topn
    else:
        pre_nms_topn_params = [6000]

    if post_nms_topn:
        post_nms_topn_params = post_nms_topn
    else:
        post_nms_topn_params = [300]

    if nms_thresh:
        nms_thresh_params = nms_thresh
    else:
        nms_thresh_params = [0.7]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, feat_stride_params, base_size_params,
                                     min_size_params,
                                     ratio_params, scale_params, pre_nms_topn_params, post_nms_topn_params,
                                     nms_thresh_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_proposal_params)


class TestProposal(object):
    # TODO: Implement Proposal layer
    @pytest.mark.skip("Not yet implemented")
    @pytest.mark.precommit
    def test_proposal_precommit(self, ie_device, precision, feat_stride, base_size, min_size, ratio, scale,
                                pre_nms_topn, post_nms_topn,
                                nms_thresh):
        self.proposal(ie_device, precision, feat_stride, base_size, min_size, ratio, scale, pre_nms_topn, post_nms_topn,
                      nms_thresh)

    # TODO: Implement Proposal layer
    @pytest.mark.skip("Not yet implemented")
    @pytest.mark.nightly
    def test_proposal_nightly(self, ie_device, precision, feat_stride, base_size, min_size, ratio, scale, pre_nms_topn,
                              post_nms_topn,
                              nms_thresh):
        self.proposal(ie_device, precision, feat_stride, base_size, min_size, ratio, scale, pre_nms_topn, post_nms_topn,
                      nms_thresh)

    def proposal(self, ie_device, precision, feat_stride, base_size, min_size, ratio, scale, pre_nms_topn,
                 post_nms_topn,
                 nms_thresh):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        proposal = network.add_layer(layer_type='Proposal',
                                     inputs=[output],
                                     feat_stride=feat_stride,
                                     base_size=base_size,
                                     min_size=min_size,
                                     ratio=ratio,
                                     scale=scale,
                                     pre_nms_topn=pre_nms_topn,
                                     post_nms_topn=post_nms_topn,
                                     nms_thresh=nms_thresh,
                                     get_out_shape_def=calc_same_out_shape,
                                     framework_representation_def=proposal_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=proposal)

        assert compare_infer_results_with_caffe(ie_results, proposal), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
