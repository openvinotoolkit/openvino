// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_prepostprocess.h"

#include <stdarg.h>

#include "common.h"

const std::map<ov_preprocess_resize_algorithm_e, ov::preprocess::ResizeAlgorithm> resize_algorithm_map = {
    {ov_preprocess_resize_algorithm_e::RESIZE_CUBIC, ov::preprocess::ResizeAlgorithm::RESIZE_CUBIC},
    {ov_preprocess_resize_algorithm_e::RESIZE_LINEAR, ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR},
    {ov_preprocess_resize_algorithm_e::RESIZE_NEAREST, ov::preprocess::ResizeAlgorithm::RESIZE_NEAREST}};

const std::map<ov_color_format_e, ov::preprocess::ColorFormat> color_format_map = {
    {ov_color_format_e::UNDEFINE, ov::preprocess::ColorFormat::UNDEFINED},
    {ov_color_format_e::NV12_SINGLE_PLANE, ov::preprocess::ColorFormat::NV12_SINGLE_PLANE},
    {ov_color_format_e::NV12_TWO_PLANES, ov::preprocess::ColorFormat::NV12_TWO_PLANES},
    {ov_color_format_e::I420_SINGLE_PLANE, ov::preprocess::ColorFormat::I420_SINGLE_PLANE},
    {ov_color_format_e::I420_THREE_PLANES, ov::preprocess::ColorFormat::I420_THREE_PLANES},
    {ov_color_format_e::RGB, ov::preprocess::ColorFormat::RGB},
    {ov_color_format_e::BGR, ov::preprocess::ColorFormat::BGR},
    {ov_color_format_e::GRAY, ov::preprocess::ColorFormat::GRAY},
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

ov_status_e ov_preprocess_prepostprocessor_get_input_info(const ov_preprocess_prepostprocessor_t* preprocess,
                                                          ov_preprocess_input_info_t** preprocess_input_info) {
    if (!preprocess || !preprocess_input_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_input_info_t> _preprocess_input_info(new ov_preprocess_input_info_t);
        _preprocess_input_info->object = &(preprocess->object->input());
        *preprocess_input_info = _preprocess_input_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_get_input_info_by_name(const ov_preprocess_prepostprocessor_t* preprocess,
                                                                  const char* tensor_name,
                                                                  ov_preprocess_input_info_t** preprocess_input_info) {
    if (!preprocess || !tensor_name || !preprocess_input_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_input_info_t> _preprocess_input_info(new ov_preprocess_input_info_t);
        _preprocess_input_info->object = &(preprocess->object->input(tensor_name));
        *preprocess_input_info = _preprocess_input_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_get_input_info_by_index(const ov_preprocess_prepostprocessor_t* preprocess,
                                                                   const size_t tensor_index,
                                                                   ov_preprocess_input_info_t** preprocess_input_info) {
    if (!preprocess || !preprocess_input_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_input_info_t> _preprocess_input_info(new ov_preprocess_input_info_t);
        _preprocess_input_info->object = &(preprocess->object->input(tensor_index));
        *preprocess_input_info = _preprocess_input_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_input_info_free(ov_preprocess_input_info_t* preprocess_input_info) {
    if (preprocess_input_info)
        delete preprocess_input_info;
}

ov_status_e ov_preprocess_input_info_get_tensor_info(const ov_preprocess_input_info_t* preprocess_input_info,
                                                     ov_preprocess_input_tensor_info_t** preprocess_input_tensor_info) {
    if (!preprocess_input_info || !preprocess_input_tensor_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_input_tensor_info_t> _preprocess_input_tensor_info(
            new ov_preprocess_input_tensor_info_t);
        _preprocess_input_tensor_info->object = &(preprocess_input_info->object->tensor());
        *preprocess_input_tensor_info = _preprocess_input_tensor_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_input_tensor_info_free(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info) {
    if (preprocess_input_tensor_info)
        delete preprocess_input_tensor_info;
}

ov_status_e ov_preprocess_input_info_get_preprocess_steps(const ov_preprocess_input_info_t* preprocess_input_info,
                                                          ov_preprocess_preprocess_steps_t** preprocess_input_steps) {
    if (!preprocess_input_info || !preprocess_input_steps) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_preprocess_steps_t> _preprocess_input_steps(new ov_preprocess_preprocess_steps_t);
        _preprocess_input_steps->object = &(preprocess_input_info->object->preprocess());
        *preprocess_input_steps = _preprocess_input_steps.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_preprocess_steps_free(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps) {
    if (preprocess_input_process_steps)
        delete preprocess_input_process_steps;
}

ov_status_e ov_preprocess_preprocess_steps_resize(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                                  const ov_preprocess_resize_algorithm_e resize_algorithm) {
    if (!preprocess_input_process_steps) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_process_steps->object->resize(resize_algorithm_map.at(resize_algorithm));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_preprocess_steps_scale(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                                 float value) {
    if (!preprocess_input_process_steps) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_process_steps->object->scale(value);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_scale_multi_channels(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                                    const float* values,
                                                    const int32_t value_size) {
    if (!preprocess_input_process_steps || !values || value_size <= 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::vector<float> scale_vec(values, values + value_size);
        preprocess_input_process_steps->object->scale(scale_vec);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_preprocess_steps_mean(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                                float value) {
    if (!preprocess_input_process_steps) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_process_steps->object->mean(value);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_mean_multi_channels(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                                   const float* values,
                                                   const int32_t value_size) {
    if (!preprocess_input_process_steps || !values || value_size <= 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::vector<float> mean_vec(values, values + value_size);
        preprocess_input_process_steps->object->mean(mean_vec);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_preprocess_steps_crop(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                                int32_t* begin,
                                                int32_t begin_size,
                                                int32_t* end,
                                                int32_t end_size) {
    if (!preprocess_input_process_steps) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::vector<int32_t> vec_begin(begin, begin + begin_size);
        std::vector<int32_t> vec_end(end, end + end_size);
        preprocess_input_process_steps->object->crop(vec_begin, vec_end);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_preprocess_steps_convert_layout(
    ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
    ov_layout_t* layout) {
    if (!preprocess_input_process_steps || !layout) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_process_steps->object->convert_layout(layout->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_preprocess_steps_reverse_channels(
    ov_preprocess_preprocess_steps_t* preprocess_input_process_steps) {
    if (!preprocess_input_process_steps) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_process_steps->object->reverse_channels();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_input_tensor_info_set_element_type(
    ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
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

ov_status_e ov_preprocess_input_tensor_info_set_from(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
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

ov_status_e ov_preprocess_input_tensor_info_set_layout(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
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

ov_status_e ov_preprocess_input_tensor_info_set_color_format_with_subname(
    ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
    const ov_color_format_e colorFormat,
    const size_t sub_names_size,
    ...) {
    if (!preprocess_input_tensor_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::vector<std::string> names = {};
        if (sub_names_size > 0) {
            va_list args_ptr;
            va_start(args_ptr, sub_names_size);
            for (size_t i = 0; i < sub_names_size; i++) {
                std::string _value = va_arg(args_ptr, char*);
                names.emplace_back(_value);
            }
            va_end(args_ptr);
        }

        preprocess_input_tensor_info->object->set_color_format(GET_OV_COLOR_FARMAT(colorFormat), names);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_input_tensor_info_set_color_format(
    ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
    const ov_color_format_e colorFormat) {
    return ov_preprocess_input_tensor_info_set_color_format_with_subname(preprocess_input_tensor_info, colorFormat, 0);
}

ov_status_e ov_preprocess_input_tensor_info_set_spatial_static_shape(
    ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
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

ov_status_e ov_preprocess_input_tensor_info_set_memory_type(
    ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
    const char* mem_type) {
    if (!preprocess_input_tensor_info || !mem_type) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        preprocess_input_tensor_info->object->set_memory_type(mem_type);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_preprocess_steps_convert_element_type(
    ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
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

ov_status_e ov_preprocess_preprocess_steps_convert_color(
    ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
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

ov_status_e ov_preprocess_prepostprocessor_get_output_info(const ov_preprocess_prepostprocessor_t* preprocess,
                                                           ov_preprocess_output_info_t** preprocess_output_info) {
    if (!preprocess || !preprocess_output_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_output_info_t> _preprocess_output_info(new ov_preprocess_output_info_t);
        _preprocess_output_info->object = &(preprocess->object->output());
        *preprocess_output_info = _preprocess_output_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_get_output_info_by_index(
    const ov_preprocess_prepostprocessor_t* preprocess,
    const size_t tensor_index,
    ov_preprocess_output_info_t** preprocess_output_info) {
    if (!preprocess || !preprocess_output_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_output_info_t> _preprocess_output_info(new ov_preprocess_output_info_t);
        _preprocess_output_info->object = &(preprocess->object->output(tensor_index));
        *preprocess_output_info = _preprocess_output_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_get_output_info_by_name(
    const ov_preprocess_prepostprocessor_t* preprocess,
    const char* tensor_name,
    ov_preprocess_output_info_t** preprocess_output_info) {
    if (!preprocess || !tensor_name || !preprocess_output_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_output_info_t> _preprocess_output_info(new ov_preprocess_output_info_t);
        _preprocess_output_info->object = &(preprocess->object->output(tensor_name));
        *preprocess_output_info = _preprocess_output_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_output_info_free(ov_preprocess_output_info_t* preprocess_output_info) {
    if (preprocess_output_info)
        delete preprocess_output_info;
}

ov_status_e ov_preprocess_output_info_get_tensor_info(
    const ov_preprocess_output_info_t* preprocess_output_info,
    ov_preprocess_output_tensor_info_t** preprocess_output_tensor_info) {
    if (!preprocess_output_info || !preprocess_output_tensor_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_output_tensor_info_t> _preprocess_output_tensor_info(
            new ov_preprocess_output_tensor_info_t);
        _preprocess_output_tensor_info->object = &(preprocess_output_info->object->tensor());
        *preprocess_output_tensor_info = _preprocess_output_tensor_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_output_tensor_info_free(ov_preprocess_output_tensor_info_t* preprocess_output_tensor_info) {
    if (preprocess_output_tensor_info)
        delete preprocess_output_tensor_info;
}

ov_status_e ov_preprocess_output_set_element_type(ov_preprocess_output_tensor_info_t* preprocess_output_tensor_info,
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

ov_status_e ov_preprocess_input_info_get_model_info(const ov_preprocess_input_info_t* preprocess_input_info,
                                                    ov_preprocess_input_model_info_t** preprocess_input_model_info) {
    if (!preprocess_input_info || !preprocess_input_model_info) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_input_model_info_t> _preprocess_input_model_info(
            new ov_preprocess_input_model_info_t);
        _preprocess_input_model_info->object = &(preprocess_input_info->object->model());
        *preprocess_input_model_info = _preprocess_input_model_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_input_model_info_free(ov_preprocess_input_model_info_t* preprocess_input_model_info) {
    if (preprocess_input_model_info)
        delete preprocess_input_model_info;
}

ov_status_e ov_preprocess_input_model_info_set_layout(ov_preprocess_input_model_info_t* preprocess_input_model_info,
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
