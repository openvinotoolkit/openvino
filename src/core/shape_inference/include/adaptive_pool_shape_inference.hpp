// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils.hpp"

namespace ov {
namespace op {
namespace adaptive_pool {

template <class TShape,
          class TOp,
          typename std::enable_if<std::is_same<TOp, v8::AdaptiveAvgPool>::value ||
                                  std::is_same<TOp, v8::AdaptiveMaxPool>::value>::type* = nullptr>
TShape out_shape_infer(const TOp* op,
                       const std::vector<TShape>& input_shapes,
                       const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    constexpr size_t spatial_dim_offset = 2;

    const auto& data_shape = input_shapes[0];
    const auto& out_spatial_shape = input_shapes[1];

    const auto& data_rank = data_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          is_rank_compatible_any_of(data_rank, {3, 4, 5}),
                          "Expected a 3D, 4D or 5D tensor for the input. Got: ",
                          data_shape);

    TShape output_shape;
    if (data_rank.is_static()) {
        auto num_of_spatial_dims = data_shape.size() - spatial_dim_offset;

        NODE_VALIDATION_CHECK(
            op,
            out_spatial_shape.rank().is_dynamic() || out_spatial_shape[0].compatible(num_of_spatial_dims),
            "Output shape for spatial dimension not compatible with data shape.");

        output_shape.reserve(data_shape.size());
        std::copy_n(data_shape.begin(), spatial_dim_offset, std::back_inserter(output_shape));

        if (const auto spatial_dims = get_input_const_data_as_shape<TShape>(op, 1, constant_data)) {
            NODE_VALIDATION_CHECK(op,
                                  num_of_spatial_dims == spatial_dims->size(),
                                  "Number of spatial dimensions is not compatible with input data rank");

            output_shape.insert(output_shape.end(), spatial_dims->begin(), spatial_dims->end());
        } else {
            output_shape.insert(output_shape.end(), num_of_spatial_dims, -1);
        }
    } else {
        output_shape = PartialShape::dynamic();
    }
    return output_shape;
}

}  // namespace adaptive_pool
}  // namespace op
}  // namespace ov
