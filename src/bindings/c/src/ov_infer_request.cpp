// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_infer_request.h"

#include "common.h"

void ov_infer_request_free(ov_infer_request_t* infer_request) {
    if (infer_request)
        delete infer_request;
}

ov_status_e ov_infer_request_set_tensor(ov_infer_request_t* infer_request,
                                        const char* tensor_name,
                                        const ov_tensor_t* tensor) {
    if (!infer_request || !tensor_name || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        infer_request->object->set_tensor(tensor_name, *tensor->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_set_tensor_by_port(ov_infer_request_t* infer_request,
                                                const ov_output_port_t* port,
                                                const ov_tensor_t* tensor) {
    if (!infer_request || !port || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        infer_request->object->set_tensor(*port->object, *tensor->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_set_tensor_by_const_port(ov_infer_request_t* infer_request,
                                                      const ov_output_const_port_t* port,
                                                      const ov_tensor_t* tensor) {
    if (!infer_request || !port || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        infer_request->object->set_tensor(*port->object, *tensor->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_set_input_tensor_by_index(ov_infer_request_t* infer_request,
                                                       const size_t idx,
                                                       const ov_tensor_t* tensor) {
    if (!infer_request || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        infer_request->object->set_input_tensor(idx, *tensor->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_set_input_tensor(ov_infer_request_t* infer_request, const ov_tensor_t* tensor) {
    if (!infer_request || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        infer_request->object->set_input_tensor(*tensor->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_set_output_tensor_by_index(ov_infer_request_t* infer_request,
                                                        const size_t idx,
                                                        const ov_tensor_t* tensor) {
    if (!infer_request || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        infer_request->object->set_output_tensor(idx, *tensor->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_set_output_tensor(ov_infer_request_t* infer_request, const ov_tensor_t* tensor) {
    if (!infer_request || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        infer_request->object->set_output_tensor(*tensor->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_tensor(const ov_infer_request_t* infer_request,
                                        const char* tensor_name,
                                        ov_tensor_t** tensor) {
    if (!infer_request || !tensor_name || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_tensor_t> _tensor(new ov_tensor_t);
        ov::Tensor tensor_get = infer_request->object->get_tensor(tensor_name);
        _tensor->object = std::make_shared<ov::Tensor>(std::move(tensor_get));
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_tensor_by_const_port(const ov_infer_request_t* infer_request,
                                                      const ov_output_const_port_t* port,
                                                      ov_tensor_t** tensor) {
    if (!infer_request || !port || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_tensor_t> _tensor(new ov_tensor_t);
        ov::Tensor tensor_get = infer_request->object->get_tensor(*port->object);
        _tensor->object = std::make_shared<ov::Tensor>(std::move(tensor_get));
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_tensor_by_port(const ov_infer_request_t* infer_request,
                                                const ov_output_port_t* port,
                                                ov_tensor_t** tensor) {
    if (!infer_request || !port || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_tensor_t> _tensor(new ov_tensor_t);
        ov::Tensor tensor_get = infer_request->object->get_tensor(*port->object);
        _tensor->object = std::make_shared<ov::Tensor>(std::move(tensor_get));
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_input_tensor_by_index(const ov_infer_request_t* infer_request,
                                                       const size_t idx,
                                                       ov_tensor_t** tensor) {
    if (!infer_request || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_tensor_t> _tensor(new ov_tensor_t);
        ov::Tensor tensor_get = infer_request->object->get_input_tensor(idx);
        _tensor->object = std::make_shared<ov::Tensor>(std::move(tensor_get));
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_input_tensor(const ov_infer_request_t* infer_request, ov_tensor_t** tensor) {
    if (!infer_request || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_tensor_t> _tensor(new ov_tensor_t);
        ov::Tensor tensor_get = infer_request->object->get_input_tensor();
        _tensor->object = std::make_shared<ov::Tensor>(std::move(tensor_get));
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_output_tensor_by_index(const ov_infer_request_t* infer_request,
                                                        const size_t idx,
                                                        ov_tensor_t** tensor) {
    if (!infer_request || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_tensor_t> _tensor(new ov_tensor_t);
        ov::Tensor tensor_get = infer_request->object->get_output_tensor(idx);
        _tensor->object = std::make_shared<ov::Tensor>(std::move(tensor_get));
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_output_tensor(const ov_infer_request_t* infer_request, ov_tensor_t** tensor) {
    if (!infer_request || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_tensor_t> _tensor(new ov_tensor_t);
        ov::Tensor tensor_get = infer_request->object->get_output_tensor();
        _tensor->object = std::make_shared<ov::Tensor>(std::move(tensor_get));
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_infer(ov_infer_request_t* infer_request) {
    if (!infer_request) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        infer_request->object->infer();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_cancel(ov_infer_request_t* infer_request) {
    if (!infer_request) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        infer_request->object->cancel();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_start_async(ov_infer_request_t* infer_request) {
    if (!infer_request) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        infer_request->object->start_async();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_wait(ov_infer_request_t* infer_request) {
    if (!infer_request) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        infer_request->object->wait();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_wait_for(ov_infer_request_t* infer_request, const int64_t timeout) {
    if (!infer_request) {
        return ov_status_e::INVALID_C_PARAM;
    }
    bool ret = true;
    try {
        ret = infer_request->object->wait_for(std::chrono::milliseconds(timeout));
    }
    CATCH_OV_EXCEPTIONS

    return ret ? ov_status_e::OK : ov_status_e::RESULT_NOT_READY;
}

ov_status_e ov_infer_request_set_callback(ov_infer_request_t* infer_request, const ov_callback_t* callback) {
    if (!infer_request || !callback) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto func = [callback](std::exception_ptr ex) {
            callback->callback_func(callback->args);
        };
        infer_request->object->set_callback(func);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_profiling_info(const ov_infer_request_t* infer_request,
                                                ov_profiling_info_list_t* profiling_infos) {
    if (!infer_request || !profiling_infos) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto infos = infer_request->object->get_profiling_info();
        size_t num = infos.size();
        profiling_infos->size = num;
        std::unique_ptr<ov_profiling_info_t[]> _profiling_info_arr(new ov_profiling_info_t[num]);
        for (size_t i = 0; i < num; i++) {
            _profiling_info_arr[i].status = (ov_profiling_info_t::Status)infos[i].status;
            _profiling_info_arr[i].real_time = infos[i].real_time.count();
            _profiling_info_arr[i].cpu_time = infos[i].cpu_time.count();

            _profiling_info_arr[i].node_name = str_to_char_array(infos[i].node_name);
            _profiling_info_arr[i].exec_type = str_to_char_array(infos[i].exec_type);
            _profiling_info_arr[i].node_type = str_to_char_array(infos[i].node_type);
        }
        profiling_infos->profiling_infos = _profiling_info_arr.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_profiling_info_list_free(ov_profiling_info_list_t* profiling_infos) {
    if (!profiling_infos) {
        return;
    }
    for (size_t i = 0; i < profiling_infos->size; i++) {
        if (profiling_infos->profiling_infos[i].node_name)
            delete[] profiling_infos->profiling_infos[i].node_name;
        if (profiling_infos->profiling_infos[i].exec_type)
            delete[] profiling_infos->profiling_infos[i].exec_type;
        if (profiling_infos->profiling_infos[i].node_type)
            delete[] profiling_infos->profiling_infos[i].node_type;
    }
    if (profiling_infos->profiling_infos)
        delete[] profiling_infos->profiling_infos;
    profiling_infos->profiling_infos = nullptr;
    profiling_infos->size = 0;
}
