def common_infer_step(device, batch, **additional_args):
    # return 'reshape' method for setting batch because old 'batch_size' method still not implemented in new API
    infer_api = {"ie_sync": {"device": device,
                             "network_modifiers": {"set_batch_using_reshape": {"batch": batch}},
                             **additional_args}}
    return "infer", infer_api


def batch_reshape_infer_step(device, batch, **additional_args):
    infer_api = {"ie_sync": {"device": device,
                             "network_modifiers": {"set_batch_using_reshape": {"batch": batch}},
                             **additional_args}}
    return "infer", infer_api


def reshape_input_shape_infer_step(device, input_file_path, **additional_args):
    infer_api = {"ie_sync": {"device": device,
                             "network_modifiers":
                                 {"reshape_input_shape": {"input_path": input_file_path}},
                             **additional_args}}
    return "infer", infer_api


def cpu_extension_infer_step(device, batch, **additional_args):
    infer_api = {"ie_sync": {"device": device, "cpu_extension": "cpu_extension",
                             "network_modifiers": {"set_batch_using_reshape": {"batch": batch}},
                             **additional_args}}

    return "infer", infer_api


def sequence_infer_step(device, batch, input_file_path=None, skip_ir_generation=False, **additional_args):
    # return 'reshape' method for setting batch because old 'batch_size' method still not implemented in new API
    infer_api = {"ie_sequence": {"device": device,
                                 "network_modifiers": {"set_batch_using_reshape": {"batch": batch}},
                                 **additional_args}}
    return "infer", infer_api
