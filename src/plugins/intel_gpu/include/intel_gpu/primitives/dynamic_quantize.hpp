// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Dynamic Quantize primitive
/// @details Performs dynamic quantization
struct dynamic_quantize : public primitive_base<dynamic_quantize> {
    CLDNN_DECLARE_PRIMITIVE(dynamic_quantize);

    dynamic_quantize() : primitive_base("", {}) {}

    /// @brief Constructs dynamic_quantize primitive
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param output_size Output data size of the primitive
    dynamic_quantize(const primitive_id& id,
           const input_info& input,
           const padding& output_padding = padding())
           : primitive_base(id, {input}, {output_padding}) {}

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        /// XXX: need to check dyn-quan params here
        return true;
    }
};
}  // namespace cldnn
