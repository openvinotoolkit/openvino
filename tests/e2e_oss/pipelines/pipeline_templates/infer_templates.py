def common_infer_step(device, batch, api_2, input_file_path=None, skip_ir_generation=False, **additional_args):
    if api_2:
        # return 'reshape' method for setting batch because old 'batch_size' method still not implemented in new API
        infer_api = {"ie_sync": {"device": device,
                                       "network_modifiers": {"set_batch_using_reshape_api_2": {"batch": batch}},
                                       **additional_args}}
    elif not api_2 and skip_ir_generation:
        infer_api = {"ie_sync": {"device": device,
                                 "network_modifiers": {"reshape_input_shape": {"input_path": input_file_path},
                                                       "set_batch": {"batch": batch}, },
                                 **additional_args}}
    else:
        infer_api = {"ie_sync": {"device": device, "network_modifiers": {"set_batch": {"batch": batch}},
                                 **additional_args}}
    return "infer", infer_api


def batch_reshape_infer_step(device, batch, api_2, **additional_args):
    if api_2:
        infer_api = {"ie_sync_api_2": {"device": device,
                                       "network_modifiers": {"set_batch_using_reshape_api_2": {"batch": batch}},
                                       **additional_args}}
    else:
        infer_api = {"ie_sync": {"device": device, "network_modifiers": {"set_batch_using_reshape": {"batch": batch}},
                                 **additional_args}}
    return "infer", infer_api


def reshape_input_shape_infer_step(device, input_file_path, api_2, **additional_args):
    if api_2:
        infer_api = {"ie_sync_api_2": {"device": device,
                                       "network_modifiers":
                                           {"reshape_input_shape_api_2": {"input_path": input_file_path}},
                                       **additional_args}}
    else:
        infer_api = {"ie_sync": {"device": device,
                                 "network_modifiers": {"reshape_input_shape": {"input_path": input_file_path}},
                                 **additional_args}}
    return "infer", infer_api


def kaldi_infer_step(device, batch, qb, device_mode, api_2, **additional_args):
    if api_2:
        # return 'reshape' method for setting batch because old 'batch_size' method was not implemented in new API
        infer_api = {"ie_speech_kaldi_api_2": {"device": device, "qb": qb, "device_mode": device_mode,
                                               "network_modifiers": {"set_batch_using_reshape_api_2": {"batch": batch}},
                                               **additional_args}}
    else:
        infer_api = {"ie_speech_kaldi": {"device": device, "qb": qb, "device_mode": device_mode,
                                         "network_modifiers": {"set_batch": {"batch": batch}},
                                         **additional_args}}
    return "infer", infer_api


def cpu_extension_infer_step(device, batch, api_2, **additional_args):
    if api_2:
        infer_api = {"ie_sync_api_2": {"device": device, "cpu_extension": "cpu_extension",
                                       "network_modifiers": {"set_batch_using_reshape_api_2": {"batch": batch}},
                                       **additional_args}}
    else:
        infer_api = {"ie_sync": {"device": device, "cpu_extension": "cpu_extension",
                                 "network_modifiers": {"set_batch": {"batch": batch}},
                                 **additional_args}}
    return "infer", infer_api


def sequence_infer_step(device, batch, api_2, input_file_path=None, skip_ir_generation=False, **additional_args):
    if api_2:
        # return 'reshape' method for setting batch because old 'batch_size' method still not implemented in new API
        infer_api = {"ie_sequence_api_2": {"device": device,
                                           "network_modifiers": {"set_batch_using_reshape_api_2": {"batch": batch}},
                                           **additional_args}}
    elif not api_2 and skip_ir_generation:
        infer_api = {"ie_sequence": {"device": device,
                                     "network_modifiers": {"reshape_input_shape": {"input_path": input_file_path},
                                                           "set_batch": {"batch": batch}, },
                                     **additional_args}}
    else:
        infer_api = {"ie_sequence": {"device": device, "network_modifiers": {"set_batch": {"batch": batch}},
                                     **additional_args}}
    return "infer", infer_api
