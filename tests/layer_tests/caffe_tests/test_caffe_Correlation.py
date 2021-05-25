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


def get_correlation_params(ie_device=None, precision=None, pad=None, kernel_size=None, max_displacement=None,
                           stride_1=None,
                           stride_2=None,
                           single_direction=None, do_abs=None, correlation_type=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if pad:
        pad_params = pad
    else:
        pad_params = range(0, 3)

    if kernel_size:
        kernel_size_params = kernel_size
    else:
        kernel_size_params = [1]

    if max_displacement:
        max_displacement_params = max_displacement
    else:
        max_displacement_params = [20]

    if stride_1:
        stride_1_params = stride_1
    else:
        stride_1_params = range(1, 3)

    if stride_2:
        stride_2_params = stride_2
    else:
        stride_2_params = range(1, 3)

    if single_direction:
        single_direction_params = single_direction
    else:
        single_direction_params = [0]

    if do_abs:
        do_abs_params = do_abs
    else:
        do_abs_params = [0.0]

    if correlation_type:
        correlation_type_params = correlation_type
    else:
        correlation_type_params = ['caffe.CorrelationParameter.MULTIPLY']

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, pad_params, kernel_size_params,
                                     max_displacement_params,
                                     stride_1_params,
                                     stride_2_params, single_direction_params, do_abs_params, correlation_type_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    # TODO: Check correctness
    """
    for device in ie_device_params:
        if device == 'GPU':
            test_args.remove(('GPU', 0))
    """
    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_correlation_params)


class TestCorrelation(object):
    @pytest.mark.precommit
    def test_correlation_precommit(self, ie_device, precision, pad, kernel_size, max_displacement, stride_1, stride_2,
                                   single_direction,
                                   do_abs,
                                   correlation_type):
        self.correlation(ie_device, precision, pad, kernel_size, max_displacement, stride_1, stride_2, single_direction,
                         do_abs,
                         correlation_type)

    @pytest.mark.nightly
    def test_correlation_nightly(self, ie_device, precision, pad, kernel_size, max_displacement, stride_1, stride_2,
                                 single_direction,
                                 do_abs,
                                 correlation_type):
        self.correlation(ie_device, precision, pad, kernel_size, max_displacement, stride_1, stride_2, single_direction,
                         do_abs,
                         correlation_type)

    def correlation(self, ie_device, precision, pad, kernel_size, max_displacement, stride_1, stride_2,
                    single_direction,
                    do_abs,
                    correlation_type):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        output2 = network.add_layer(layer_type='Input',
                                    get_out_shape_def=calc_out_shape_input_layer,
                                    framework_representation_def=input_to_proto)
        corr = network.add_layer(layer_type='Correlation',
                                 inputs=[output, output2],
                                 pad=pad, kernel_size=kernel_size,
                                 max_displacement=max_displacement, stride_1=stride_1, stride_2=stride_2,
                                 single_direction=single_direction, do_abs=do_abs, correlation_type=correlation_type,
                                 get_out_shape_def=calc_out_shape_correlation,
                                 framework_representation_def=correlation_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=corr.name)

        assert compare_infer_results_with_caffe(ie_results, corr.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
