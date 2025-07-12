// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"
#include "ov_ops/msda.hpp"
#include <vector>

namespace cldnn {

using MSDA = ov::op::internal::MSDA;

/// @brief msda primitive
/// @details Performs MSDA
struct msda : public primitive_base<msda> {
    CLDNN_DECLARE_PRIMITIVE(msda)

    msda() : primitive_base("", {}) {}

    /// @brief Constructs msda primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    msda(const primitive_id& id,
            const std::vector<input_info>& inputs)
        : primitive_base(id, inputs) {
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<msda>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<msda>::load(ib);
    }
};

}  // namespace cldnn