// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/tile.hpp>

#include "compare.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const Tile* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    using TDim = typename std::iterator_traits<typename T::iterator>::value_type;

    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);

    const auto& repeats_shape = input_shapes[1];
    NODE_VALIDATION_CHECK(op, repeats_shape.rank().compatible(1), "Tile repeats must be of rank 1");

    const auto& arg_shape = input_shapes[0];
    auto& output_shape = output_shapes[0];

    // Get repeats
    std::vector<int64_t> repeats;
    bool has_repeats = get_data_as_int64<T>(1, op, repeats, constant_data);

    const auto& arg_rank = arg_shape.rank();
    if (arg_rank.is_static() && has_repeats) {
        const auto output_rank = std::max(arg_shape.size(), repeats.size());

        std::vector<TDim> dims;
        dims.reserve(output_rank);

        // add missing repeats and set negatives to 0
        auto rep_it = repeats.insert(repeats.begin(), output_rank - repeats.size(), 1);
        std::replace_if(rep_it, repeats.end(), cmp::Less<int64_t>(0), 0);

        // insert missing input dimensions
        rep_it = std::next(repeats.begin(), output_rank - arg_shape.size());
        dims.insert(dims.begin(), repeats.begin(), rep_it);

        // calc repeated output dimensions
        std::transform(arg_shape.begin(), arg_shape.end(), rep_it, std::back_inserter(dims), std::multiplies<TDim>());

        output_shape = T(std::move(dims));
    } else if (arg_rank.is_static() && repeats_shape[0].is_static()) {
        // unknown repeats but shape is 1-D static, any dim can be repeated (add missing dimension)
        output_shape.resize(std::max<size_t>(arg_rank.get_length(), repeats_shape[0].get_length()));
    } else {
        // can't deduce shape, set default value
        output_shape = PartialShape::dynamic();
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
