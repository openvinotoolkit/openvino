// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/tile.hpp>

#include "shape_infer_type_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v0 {

template <class TShape>
struct NegativeToZero {
    NegativeToZero() = default;
    template <class U>
    TShape operator()(const U u) const {
        return static_cast<TShape>(std::max<U>(0, ov::util::InTypeRange<U>()(u)));
    }
};

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Tile* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    using TDim = typename TShape::value_type;
    using TDimValue = typename TDim::value_type;

    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    const auto& repeats_shape = input_shapes[1];
    const auto& repeats_rank = repeats_shape.rank();
    NODE_VALIDATION_CHECK(op, repeats_rank.compatible(1), "Tile repeats must be of rank 1");

    const auto& arg_shape = input_shapes[0];
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];

    // Get repeats and pre process values
    constexpr auto negative_repeats_to_zero = NegativeToZero<TDimValue>();

    auto repeats = get_input_const_data_as_shape<TRShape>(op, 1, tensor_accessor, negative_repeats_to_zero);

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
    return output_shapes;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
