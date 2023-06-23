// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transposition_info.hpp"

#include <vector>

#include "common/graph_utils.hpp"
#include "log/debug.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"

namespace ov {
namespace intel_gna {
namespace pre_post_processing {

using namespace ov::opset10;

std::shared_ptr<ov::Model> ToProcessModel(const TranspositionInfo& t_info) {
    int32_t c_size = static_cast<int32_t>(t_info.num_transpose_rows);
    int32_t hw_size = static_cast<int32_t>(t_info.num_transpose_columns);

    if (!t_info.transpose) {
        return nullptr;
    }

    ov::PartialShape input_shape{1, c_size, hw_size};
    auto param = std::make_shared<Parameter>(ov::element::f32, input_shape);

    // legacy way was to swap C and HW dimensions in the reshaped tensor
    std::vector<int32_t> reshape_pattern{-1, c_size, hw_size};
    auto reshape_const =
        std::make_shared<Constant>(ov::element::i32, ov::Shape{reshape_pattern.size()}, reshape_pattern);
    auto reshape = std::make_shared<Reshape>(param, reshape_const, false);

    // CHW -> HWC or HWC -> CHW
    std::vector<int8_t> transpose_order{0, 2, 1};
    auto transpose_const =
        std::make_shared<Constant>(ov::element::i8, ov::Shape{transpose_order.size()}, transpose_order);
    auto transpose = std::make_shared<Transpose>(reshape, transpose_const);

    auto result = std::make_shared<Result>(transpose);

    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    return model;
}

std::shared_ptr<ov::Model> ToProcessModel(const std::vector<TranspositionInfo>& transposes) {
    // count transposition parts need to be transposed
    auto count_transposes = std::count_if(transposes.begin(), transposes.end(), [](TranspositionInfo t_info) {
        return t_info.transpose || t_info.num_transpose_rows != 1 || t_info.num_transpose_columns != 1;
    });
    if (count_transposes == 0) {
        return nullptr;
    }

    // case when the input should be transposed entirely
    if (transposes.size() == 1) {
        return ToProcessModel(transposes.front());
    }

    std::vector<size_t> indices = {};
    for (auto& transpose : transposes) {
        size_t c_size = transpose.num_transpose_rows;
        size_t hw_size = transpose.num_transpose_columns;
        if (c_size == 0 || hw_size == 0) {
            THROW_GNA_EXCEPTION << "Incorrect transposition dimentions";
        }
        std::vector<size_t> slice_indices;
        std::vector<size_t> transpose_order =
            transpose.transpose ? std::vector<size_t>{1, 0} : std::vector<size_t>{0, 1};
        slice_indices =
            graph_utils::make_gather_indexes_from_transpose_axes(ov::Shape{c_size, hw_size}, transpose_order);
        size_t id = indices.size();
        std::for_each(slice_indices.begin(), slice_indices.end(), [&id](size_t& i) {
            i += id;
        });
        indices.insert(indices.end(), slice_indices.begin(), slice_indices.end());
    }

    auto param = std::make_shared<Parameter>(ov::element::f32, ov::Shape{1, indices.size()});
    // legacy way was to swap C and HW dimensions in the reshaped tensor
    std::vector<int32_t> reshape_pattern{-1, static_cast<int32_t>(indices.size())};
    auto reshape_const =
        std::make_shared<Constant>(ov::element::i32, ov::Shape{reshape_pattern.size()}, reshape_pattern);
    auto reshape = std::make_shared<Reshape>(param, reshape_const, false);

    // CHW -> HWC or HWC -> CHW
    auto gather_indices = std::make_shared<Constant>(ov::element::i32, ov::Shape{indices.size()}, indices);
    auto gather_axis = std::make_shared<Constant>(ov::element::i8, ov::Shape{1}, std::vector<int8_t>{1});
    auto gather = std::make_shared<Gather>(reshape, gather_indices, gather_axis);

    auto result = std::make_shared<Result>(gather);

    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    return model;
}

}  // namespace pre_post_processing
}  // namespace intel_gna
}  // namespace ov
