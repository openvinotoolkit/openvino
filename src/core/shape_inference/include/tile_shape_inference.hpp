// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/tile.hpp>

#include "shape_infer_type_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v0 {

template <class T>
std::vector<T> shape_infer(const Tile* op,
                           const std::vector<T>& input_shapes,
                           const std::map<size_t, std::reference_wrapper<const ov::Tensor>>& constant_data = {}) {
    using TDim = typename T::value_type;
    using TDimValue = typename TDim::value_type;

    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    const auto& repeats_shape = input_shapes[1];
    const auto& repeats_rank = repeats_shape.rank();
    NODE_VALIDATION_CHECK(op, repeats_rank.compatible(1), "Tile repeats must be of rank 1");

    const auto& arg_shape = input_shapes[0];
    T output_shape;

    // Get repeats and pre process values
    auto negative_repeats_to_zero = [](const TDimValue v) -> TDimValue {
        return std::max<TDimValue>(0, ov::util::InTypeRange<TDimValue>()(v));
    };

    auto repeats = get_input_const_data_as_shape<T>(op, 1, constant_data, negative_repeats_to_zero);

    const auto& arg_rank = arg_shape.rank();
    if (arg_rank.is_static() && repeats) {
        const auto output_rank = std::max(arg_shape.size(), repeats->size());
        output_shape.reserve(output_rank);

        // add missing repeats
        repeats->insert(repeats->begin(), output_rank - repeats->size(), TDim{1});

        // insert missing input dimensions
        auto rep_it = std::next(repeats->begin(), output_rank - arg_shape.size());
        output_shape.insert(output_shape.begin(), repeats->begin(), rep_it);

        // calc repeated output dimensions
        std::transform(arg_shape.begin(),
                       arg_shape.end(),
                       rep_it,
                       std::back_inserter(output_shape),
                       std::multiplies<TDim>());
    } else if (arg_rank.is_static() && repeats_rank.is_static() && repeats_shape[0].is_static()) {
        // unknown repeats any dim can be repeated (add missing dimension)
        output_shape.resize(std::max<size_t>(arg_rank.get_length(), repeats_shape[0].get_length()));
    } else {
        // can't deduce shape, set default value
        output_shape = PartialShape::dynamic();
    }
    return {output_shape};
}
}  // namespace v0
}  // namespace op
}  // namespace ov
