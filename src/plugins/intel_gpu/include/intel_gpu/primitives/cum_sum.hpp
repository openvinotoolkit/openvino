// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {


struct cum_sum : public primitive_base<cum_sum> {
    CLDNN_DECLARE_PRIMITIVE(cum_sum)

    /// @brief Constructs cum_sum primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param axis Scalar axis.
    /// @param exclusive If set to true then the top element is not included in sum.
    /// @param reverse If set to true will perform the sums in reverse direction.
    cum_sum(const primitive_id& id,
            const input_info& input,
            const int64_t axis = 0,
            const bool exclusive = false,
            const bool reverse = false,
            const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}), axis(axis), exclusive(exclusive), reverse(reverse)
    {}

    /// @brief Scalar axis.
    int64_t axis;
    /// @brief If set to true then the top element is not included in sum.
    bool exclusive;
    /// @brief If set to true will perform the sums in reverse direction.
    bool reverse;
};
}  // namespace cldnn
