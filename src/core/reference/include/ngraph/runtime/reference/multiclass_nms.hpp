// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <ngraph/runtime/host_tensor.hpp>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/util/multiclass_nms_base.hpp"
#include "ngraph/op/util/nms_base.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
namespace multiclass_nms_impl {
namespace {
std::vector<float> get_floats(const std::shared_ptr<HostTensor>& input, const Shape& shape) {
    size_t input_size = shape_size(shape);
    std::vector<float> result(input_size);

    switch (input->get_element_type()) {
    case element::Type_t::bf16: {
        bfloat16* p = input->get_data_ptr<bfloat16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case element::Type_t::f16: {
        float16* p = input->get_data_ptr<float16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case element::Type_t::f32: {
        float* p = input->get_data_ptr<float>();
        memcpy(result.data(), p, input_size * sizeof(float));
    } break;
    default:
        throw std::runtime_error("Unsupported data type.");
        break;
    }

    return result;
}

std::vector<int64_t> get_integers(const std::shared_ptr<HostTensor>& input, const Shape& shape) {
    size_t input_size = shape_size(shape);
    std::vector<int64_t> result(input_size);

    switch (input->get_element_type()) {
    case element::Type_t::i8: {
        auto p = input->get_data_ptr<int8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::i16: {
        auto p = input->get_data_ptr<int16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::i32: {
        auto p = input->get_data_ptr<int32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::i64: {
        auto p = input->get_data_ptr<int64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u8: {
        auto p = input->get_data_ptr<uint8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u16: {
        auto p = input->get_data_ptr<uint16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u32: {
        auto p = input->get_data_ptr<uint32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u64: {
        auto p = input->get_data_ptr<uint64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    default:
        throw std::runtime_error("Unsupported data type in op NonMaxSuppression-5");
        break;
    }

    return result;
}
}  // namespace

struct InfoForNMS {
    Shape selected_outputs_shape;
    Shape selected_indices_shape;
    Shape selected_numrois_shape;
    Shape boxes_shape;
    Shape scores_shape;
    Shape roisnum_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    std::vector<int64_t> roisnum_data;
    size_t selected_outputs_shape_size;
    size_t selected_indices_shape_size;
    size_t selected_numrois_shape_size;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;
constexpr size_t roisnum_port = 2;

inline PartialShape infer_selected_outputs_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                                 int nms_top_k,
                                                 int keep_top_k) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    PartialShape result = {Dimension::dynamic(), 6};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() &&
            scores_ps[1].is_static()) {  // FIXME: Do we need this check here?
            const bool shared = (scores_ps.rank().get_length() == 3);

            const auto num_boxes = shared ? boxes_ps[1].get_length() : boxes_ps[0].get_length();
            const auto num_classes = shared ? scores_ps[1].get_length() : boxes_ps[1].get_length();
            auto num_images = scores_ps[0].get_length();
            if (!shared) {
                const auto roisnum_ps = inputs[roisnum_port]->get_partial_shape();
                num_images = roisnum_ps[0].get_length();
            }

            int64_t max_output_boxes_per_class = 0;
            if (nms_top_k >= 0)
                max_output_boxes_per_class = std::min(num_boxes, (int64_t)nms_top_k);
            else
                max_output_boxes_per_class = num_boxes;

            auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
            if (keep_top_k >= 0)
                max_output_boxes_per_batch = std::min(max_output_boxes_per_batch, (int64_t)keep_top_k);

            result[0] = max_output_boxes_per_batch * num_images;
        }
    }

    return result;
}

inline std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes, const Shape& boxes_shape) {
    auto result = get_floats(boxes, boxes_shape);
    return result;
}

inline std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores, const Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

inline std::vector<int64_t> prepare_roisnum_data(const std::shared_ptr<HostTensor>& roisnum,
                                                 const Shape& roisnum_shape) {
    auto result = get_integers(roisnum, roisnum_shape);
    return result;
}

inline InfoForNMS get_info_for_nms_eval(const std::shared_ptr<op::util::MulticlassNmsBase>& nms,
                                        const std::vector<std::shared_ptr<HostTensor>>& inputs) {
    InfoForNMS result;

    auto selected_outputs_shape = infer_selected_outputs_shape(inputs, nms->get_nms_top_k(), nms->get_keep_top_k());
    result.selected_outputs_shape = selected_outputs_shape.to_shape();
    result.selected_indices_shape = {result.selected_outputs_shape[0], 1};

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape);
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    if (inputs.size() == 3) {
        result.roisnum_shape = inputs[roisnum_port]->get_shape();
        result.roisnum_data = prepare_roisnum_data(inputs[roisnum_port], result.roisnum_shape);
        result.selected_numrois_shape = {result.roisnum_shape[0]};
    } else {
        result.selected_numrois_shape = {inputs[0]->get_shape()[0]};
    }

    result.selected_outputs_shape_size = shape_size(result.selected_outputs_shape);
    result.selected_indices_shape_size = shape_size(result.selected_indices_shape);
    result.selected_numrois_shape_size = shape_size(result.selected_numrois_shape);

    return result;
}
}  // namespace multiclass_nms_impl

void multiclass_nms(const float* boxes_data,
                    const Shape& boxes_data_shape,
                    const float* scores_data,
                    const Shape& scores_data_shape,
                    const int64_t* roisnum_data,
                    const Shape& roisnum_data_shape,
                    const op::util::MulticlassNmsBase::Attributes& attrs,
                    float* selected_outputs,
                    const Shape& selected_outputs_shape,
                    int64_t* selected_indices,
                    const Shape& selected_indices_shape,
                    int64_t* valid_outputs);

}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
