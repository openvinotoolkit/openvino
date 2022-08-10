// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_prepostprocess.h"

#include "common.h"

const std::map<ov_preprocess_resizealgorithm_e, ov::preprocess::ResizeAlgorithm> resize_algorithm_map = {
    {ov_preprocess_resizealgorithm_e::RESIZE_CUBIC, ov::preprocess::ResizeAlgorithm::RESIZE_CUBIC},
    {ov_preprocess_resizealgorithm_e::RESIZE_LINEAR, ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR},
    {ov_preprocess_resizealgorithm_e::RESIZE_NEAREST, ov::preprocess::ResizeAlgorithm::RESIZE_NEAREST}};

const std::map<ov_color_format_e, ov::preprocess::ColorFormat> color_format_map = {
    {ov_color_format_e::UNDEFINE, ov::preprocess::ColorFormat::UNDEFINED},
    {ov_color_format_e::NV12_SINGLE_PLANE, ov::preprocess::ColorFormat::NV12_SINGLE_PLANE},
    {ov_color_format_e::NV12_TWO_PLANES, ov::preprocess::ColorFormat::NV12_TWO_PLANES},
    {ov_color_format_e::I420_SINGLE_PLANE, ov::preprocess::ColorFormat::I420_SINGLE_PLANE},
    {ov_color_format_e::I420_THREE_PLANES, ov::preprocess::ColorFormat::I420_THREE_PLANES},
    {ov_color_format_e::RGB, ov::preprocess::ColorFormat::RGB},
    {ov_color_format_e::BGR, ov::preprocess::ColorFormat::BGR},
    {ov_color_format_e::RGBX, ov::preprocess::ColorFormat::RGBX},
    {ov_color_format_e::BGRX, ov::preprocess::ColorFormat::BGRX}};

#define GET_OV_COLOR_FARMAT(a)                                                                   \
    (color_format_map.find(a) == color_format_map.end() ? ov::preprocess::ColorFormat::UNDEFINED \
                                                        : color_format_map.at(a))

ov_status_e ov_preprocess_prepostprocessor_create(const ov_model_t* model,
                                                  ov_preprocess_prepostprocessor_t** preprocess) {
    if (!model || !preprocess) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_prepostprocessor_t> _preprocess(new ov_preprocess_prepostprocessor_t);
        _preprocess->object = std::make_shared<ov::preprocess::PrePostProcessor>(model->object);
        *preprocess = _preprocess.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_prepostprocessor_free(ov_preprocess_prepostprocessor_t* preprocess) {
    if (preprocess)
        delete preprocess;
}

ov_status_e ov_preprocess_prepostprocessor_input(const ov_preprocess_prepostprocessor_t* preprocess,
                                                 ov_preprocess_inputinfo_t** preprocess_input_info) {
    if (!preprocess || !preprocess_input_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_inputinfo_t> _preprocess_input_info(new ov_preprocess_inputinfo_t);
        _preprocess_input_info->object = &(preprocess->object->input());
        *preprocess_input_info = _preprocess_input_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_input_by_name(const ov_preprocess_prepostprocessor_t* preprocess,
                                                         const char* tensor_name,
                                                         ov_preprocess_inputinfo_t** preprocess_input_info) {
    if (!preprocess || !tensor_name || !preprocess_input_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_inputinfo_t> _preprocess_input_info(new ov_preprocess_inputinfo_t);
        _preprocess_input_info->object = &(preprocess->object->input(tensor_name));
        *preprocess_input_info = _preprocess_input_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_input_by_index(const ov_preprocess_prepostprocessor_t* preprocess,
                                                          const size_t tensor_index,
                                                          ov_preprocess_inputinfo_t** preprocess_input_info) {
    if (!preprocess || !preprocess_input_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_inputinfo_t> _preprocess_input_info(new ov_preprocess_inputinfo_t);
        _preprocess_input_info->object = &(preprocess->object->input(tensor_index));
        *preprocess_input_info = _preprocess_input_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_inputinfo_free(ov_preprocess_inputinfo_t* preprocess_input_info) {
    if (preprocess_input_info)
        delete preprocess_input_info;
}

ov_status_e ov_preprocess_inputinfo_tensor(const ov_preprocess_inputinfo_t* preprocess_input_info,
                                           ov_preprocess_inputtensorinfo_t** preprocess_input_tensor_info) {
    if (!preprocess_input_info || !preprocess_input_tensor_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_inputtensorinfo_t> _preprocess_input_tensor_info(
            new ov_preprocess_inputtensorinfo_t);
        _preprocess_input_tensor_info->object = &(preprocess_input_info->object->tensor());
        *preprocess_input_tensor_info = _preprocess_input_tensor_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_inputtensorinfo_free(ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info) {
    if (preprocess_input_tensor_info)
        delete preprocess_input_tensor_info;
}

ov_status_e ov_preprocess_inputinfo_preprocess(const ov_preprocess_inputinfo_t* preprocess_input_info,
                                               ov_preprocess_preprocesssteps_t** preprocess_input_steps) {
    if (!preprocess_input_info || !preprocess_input_steps) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_preprocesssteps_t> _preprocess_input_steps(new ov_preprocess_preprocesssteps_t);
        _preprocess_input_steps->object = &(preprocess_input_info->object->preprocess());
        *preprocess_input_steps = _preprocess_input_steps.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_preprocesssteps_free(ov_preprocess_preprocesssteps_t* preprocess_input_process_steps) {
    if (preprocess_input_process_steps)
        delete preprocess_input_process_steps;
}

ov_status_e ov_preprocess_preprocesssteps_resize(ov_preprocess_preprocesssteps_t* preprocess_input_process_steps,
                                                 const ov_preprocess_resizealgorithm_e resize_algorithm) {
    if (!preprocess_input_process_steps) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_process_steps->object->resize(resize_algorithm_map.at(resize_algorithm));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_inputtensorinfo_set_element_type(
    ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
    const ov_element_type_e element_type) {
    if (!preprocess_input_tensor_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_tensor_info->object->set_element_type(get_element_type(element_type));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_inputtensorinfo_set_from(ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
                                                   const ov_tensor_t* tensor) {
    if (!preprocess_input_tensor_info || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_tensor_info->object->set_from(*(tensor->object));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_inputtensorinfo_set_layout(ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
                                                     ov_layout_t* layout) {
    if (!preprocess_input_tensor_info || !layout) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_tensor_info->object->set_layout(layout->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_inputtensorinfo_set_color_format(
    ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
    const ov_color_format_e colorFormat) {
    if (!preprocess_input_tensor_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_tensor_info->object->set_color_format(GET_OV_COLOR_FARMAT(colorFormat));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_inputtensorinfo_set_spatial_static_shape(
    ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
    const size_t input_height,
    const size_t input_width) {
    if (!preprocess_input_tensor_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_tensor_info->object->set_spatial_static_shape(input_height, input_width);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_preprocesssteps_convert_element_type(
    ov_preprocess_preprocesssteps_t* preprocess_input_process_steps,
    const ov_element_type_e element_type) {
    if (!preprocess_input_process_steps) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_process_steps->object->convert_element_type(get_element_type(element_type));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_preprocesssteps_convert_color(ov_preprocess_preprocesssteps_t* preprocess_input_process_steps,
                                                        const ov_color_format_e colorFormat) {
    if (!preprocess_input_process_steps) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_process_steps->object->convert_color(GET_OV_COLOR_FARMAT(colorFormat));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_output(const ov_preprocess_prepostprocessor_t* preprocess,
                                                  ov_preprocess_outputinfo_t** preprocess_output_info) {
    if (!preprocess || !preprocess_output_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_outputinfo_t> _preprocess_output_info(new ov_preprocess_outputinfo_t);
        _preprocess_output_info->object = &(preprocess->object->output());
        *preprocess_output_info = _preprocess_output_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_output_by_index(const ov_preprocess_prepostprocessor_t* preprocess,
                                                           const size_t tensor_index,
                                                           ov_preprocess_outputinfo_t** preprocess_output_info) {
    if (!preprocess || !preprocess_output_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_outputinfo_t> _preprocess_output_info(new ov_preprocess_outputinfo_t);
        _preprocess_output_info->object = &(preprocess->object->output(tensor_index));
        *preprocess_output_info = _preprocess_output_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_output_by_name(const ov_preprocess_prepostprocessor_t* preprocess,
                                                          const char* tensor_name,
                                                          ov_preprocess_outputinfo_t** preprocess_output_info) {
    if (!preprocess || !tensor_name || !preprocess_output_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_outputinfo_t> _preprocess_output_info(new ov_preprocess_outputinfo_t);
        _preprocess_output_info->object = &(preprocess->object->output(tensor_name));
        *preprocess_output_info = _preprocess_output_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_outputinfo_free(ov_preprocess_outputinfo_t* preprocess_output_info) {
    if (preprocess_output_info)
        delete preprocess_output_info;
}

ov_status_e ov_preprocess_outputinfo_tensor(ov_preprocess_outputinfo_t* preprocess_output_info,
                                            ov_preprocess_outputtensorinfo_t** preprocess_output_tensor_info) {
    if (!preprocess_output_info || !preprocess_output_tensor_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_outputtensorinfo_t> _preprocess_output_tensor_info(
            new ov_preprocess_outputtensorinfo_t);
        _preprocess_output_tensor_info->object = &(preprocess_output_info->object->tensor());
        *preprocess_output_tensor_info = _preprocess_output_tensor_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_outputtensorinfo_free(ov_preprocess_outputtensorinfo_t* preprocess_output_tensor_info) {
    if (preprocess_output_tensor_info)
        delete preprocess_output_tensor_info;
}

ov_status_e ov_preprocess_output_set_element_type(ov_preprocess_outputtensorinfo_t* preprocess_output_tensor_info,
                                                  const ov_element_type_e element_type) {
    if (!preprocess_output_tensor_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_output_tensor_info->object->set_element_type(get_element_type(element_type));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_inputinfo_model(ov_preprocess_inputinfo_t* preprocess_input_info,
                                          ov_preprocess_inputmodelinfo_t** preprocess_input_model_info) {
    if (!preprocess_input_info || !preprocess_input_model_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_inputmodelinfo_t> _preprocess_input_model_info(
            new ov_preprocess_inputmodelinfo_t);
        _preprocess_input_model_info->object = &(preprocess_input_info->object->model());
        *preprocess_input_model_info = _preprocess_input_model_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_inputmodelinfo_free(ov_preprocess_inputmodelinfo_t* preprocess_input_model_info) {
    if (preprocess_input_model_info)
        delete preprocess_input_model_info;
}

ov_status_e ov_preprocess_inputmodelinfo_set_layout(ov_preprocess_inputmodelinfo_t* preprocess_input_model_info,
                                                    ov_layout_t* layout) {
    if (!preprocess_input_model_info || !layout) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_model_info->object->set_layout(layout->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_build(const ov_preprocess_prepostprocessor_t* preprocess,
                                                 ov_model_t** model) {
    if (!preprocess || !model) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_model_t> _model(new ov_model_t);
        _model->object = preprocess->object->build();
        *model = _model.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}
