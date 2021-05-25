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


def get_data_augmentation_params(ie_device=None, precision=None, crop_width=None, crop_height=None, write=None,
                                 max_multiplier=None,
                                 augment_during_test=None, recompute_mean=None, write_mean=None, mean_per_pixel=None,
                                 bottomwidth=None, bottomheight=None, num=None):
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if crop_width:
        crop_width_params = crop_width
    else:
        crop_width_params = [100]

    if crop_height:
        crop_height_params = crop_height
    else:
        crop_height_params = [100]

    if write:
        write_augmented_params = write
    else:
        write_augmented_params = ['']

    if max_multiplier:
        max_multiplier_params = max_multiplier
    else:
        max_multiplier_params = [0.1, 255]

    if augment_during_test:
        augment_during_test_params = augment_during_test
    else:
        augment_during_test_params = [True, False]

    if recompute_mean:
        recompute_mean_params = recompute_mean
    else:
        recompute_mean_params = [100]

    if write_mean:
        write_mean_params = write_mean
    else:
        write_mean_params = ['']

    if mean_per_pixel:
        mean_per_pixel_params = mean_per_pixel
    else:
        mean_per_pixel_params = [True, False]

    if bottomwidth:
        bottomwidth_params = bottomwidth
    else:
        bottomwidth_params = [100, 0]

    if bottomheight:
        bottomheight_params = bottomheight
    else:
        bottomheight_params = [100]

    if num:
        num_params = num
    else:
        num_params = [1]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, crop_width_params, crop_height_params,
                                     write_augmented_params,
                                     max_multiplier_params, augment_during_test_params, recompute_mean_params,
                                     write_mean_params, mean_per_pixel_params, bottomwidth_params, bottomheight_params,
                                     num_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_data_augmentation_params)


class TestDataAugmentation(object):
    @pytest.mark.precommit
    def test_data_augmentation_precommit(self, ie_device, precision, crop_width, crop_height, write, max_multiplier,
                                         augment_during_test, recompute_mean, write_mean, mean_per_pixel, bottomwidth,
                                         bottomheight, num):
        self.data_augmentation(ie_device, precision, crop_width, crop_height, write, max_multiplier,
                               augment_during_test, recompute_mean, write_mean, mean_per_pixel, bottomwidth,
                               bottomheight, num)

    @pytest.mark.nightly
    def test_data_augmentation_nightly(self, ie_device, precision, crop_width, crop_height, write, max_multiplier,
                                       augment_during_test, recompute_mean, write_mean, mean_per_pixel, bottomwidth,
                                       bottomheight, num):
        self.data_augmentation(ie_device, precision, crop_width, crop_height, write, max_multiplier,
                               augment_during_test, recompute_mean, write_mean, mean_per_pixel, bottomwidth,
                               bottomheight, num)

    def data_augmentation(self, ie_device, precision, crop_width, crop_height, write, max_multiplier,
                          augment_during_test, recompute_mean, write_mean, mean_per_pixel, bottomwidth, bottomheight,
                          num):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        aug = network.add_layer(layer_type='DataAugmentation',
                                inputs=[output],
                                crop_width=crop_width,
                                crop_height=crop_height,
                                max_multiplier=max_multiplier,
                                augment_during_test=augment_during_test,
                                recompute_mean=recompute_mean,
                                mean_per_pixel=mean_per_pixel,
                                bottomwidth=bottomwidth,
                                bottomheight=bottomheight,
                                num=num,
                                get_out_shape_def=caffe_calc_data_augmentation_layer,
                                framework_representation_def=data_augmentation_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=aug.name)

        assert compare_infer_results_with_caffe(ie_results, aug.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
