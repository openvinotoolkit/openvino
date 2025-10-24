// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "primitive.hpp"

namespace cldnn {
using MOECompressed = ov::intel_gpu::op::MOECompressed;

/// @brief moe compressed primitive
/// @details Performs moe compressed
struct moe_compressed : public primitive_base<moe_compressed> {
    CLDNN_DECLARE_PRIMITIVE(moe_compressed)

    moe_compressed() : primitive_base("", {}) {}

    /// @brief Constructs moe primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    moe_compressed(const primitive_id& id, const std::vector<input_info>& inputs, const MOECompressed::Config& config)
        : primitive_base(id, inputs, 1, {optional_data_type()}),
          _config(config) {}

    MOECompressed::Config _config;

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const moe_compressed>(rhs);

        return std::memcmp(&_config, &rhs_casted._config, sizeof(_config)) == 0;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_compressed>::save(ob);
        ob << make_data(&_config, sizeof(_config));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_compressed>::load(ib);
        ib >> make_data(&_config, sizeof(_config));
    }
};

}  // namespace cldnn
