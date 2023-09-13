// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/runtime/memory.hpp"

namespace cldnn {

/// @brief Provides input data to topology.
/// @details This primitive allows to pass data which is known at topology creation.
/// For example, weights and biases for scoring networks.
/// @note Passing data at topology may improve network performance if data optimization is enabled.
struct data : public primitive_base<data> {
    CLDNN_DECLARE_PRIMITIVE(data)

    data() : primitive_base("", {}) {}

    /// @brief Constructs data primitive.
    /// @param id This primitive id.
    /// @param mem @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    data(const primitive_id& id, memory::ptr mem)
        : primitive_base(id, {}, {padding()}), mem(std::move(mem)) {}

    /// @brief @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    memory::ptr mem;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, id);
        return seed;
    }
};
}  // namespace cldnn
