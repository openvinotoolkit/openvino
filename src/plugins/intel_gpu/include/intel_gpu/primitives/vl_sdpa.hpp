// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"
#include "ov_ops/vl_sdpa.hpp"
#include <vector>

namespace cldnn {

using VLSDPA = ov::op::internal::VLSDPA;

/// @brief vl_sdpa primitive
/// @details Performs VL SDPA
struct vl_sdpa : public primitive_base<vl_sdpa> {
    CLDNN_DECLARE_PRIMITIVE(vl_sdpa)

    vl_sdpa() : primitive_base("", {}) {}

    /// @brief Constructs vl_sdpa primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    vl_sdpa(const primitive_id& id,
            const std::vector<input_info>& inputs,
            const std::vector<int64_t>& input_q_transpose_order = {},
            const std::vector<int64_t>& input_k_transpose_order = {},
            const std::vector<int64_t>& input_v_transpose_order = {},
            const std::vector<int64_t>& output_transpose_order = {})
        : primitive_base(id, inputs)
        , input_q_transpose_order(input_q_transpose_order)
        , input_k_transpose_order(input_k_transpose_order)
        , input_v_transpose_order(input_v_transpose_order)
        , output_transpose_order(output_transpose_order) {
    }

    std::vector<int64_t> input_q_transpose_order;
    std::vector<int64_t> input_k_transpose_order;
    std::vector<int64_t> input_v_transpose_order;
    std::vector<int64_t> output_transpose_order;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, input_q_transpose_order.begin(), input_q_transpose_order.end());
        seed = hash_range(seed, input_k_transpose_order.begin(), input_k_transpose_order.end());
        seed = hash_range(seed, input_v_transpose_order.begin(), input_v_transpose_order.end());
        seed = hash_range(seed, output_transpose_order.begin(), output_transpose_order.end());

        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<vl_sdpa>::save(ob);
        ob << input_q_transpose_order;
        ob << input_k_transpose_order;
        ob << input_v_transpose_order;
        ob << output_transpose_order;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<vl_sdpa>::load(ib);
        ib >> input_q_transpose_order;
        ib >> input_k_transpose_order;
        ib >> input_v_transpose_order;
        ib >> output_transpose_order;
    }
};

}  // namespace cldnn
