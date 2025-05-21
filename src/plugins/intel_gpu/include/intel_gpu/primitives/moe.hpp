// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/graph/serialization/memory_serializer.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "primitive.hpp"
#include "ov_ops/moe.hpp"
#include <vector>

namespace cldnn {
using MOE = ov::op::internal::MOE;

struct mlp_params {
    struct param {
        cldnn::memory::ptr weight;
        cldnn::memory::ptr bias;
        cldnn::memory::ptr scale;
        cldnn::memory::ptr zp;
        // ba layout copy
        cldnn::memory::ptr scale_ba;
        cldnn::memory::ptr zp_ba;
        bool operator==(const param& rhs) const {
            return weight == rhs.weight && bias == rhs.bias && scale == rhs.scale && zp == rhs.zp;
        }
    } param[3];

    bool operator==(const mlp_params& rhs) const {
        return param[0] == rhs.param[0] && param[1] == rhs.param[1] && param[2] == rhs.param[2];
    }
};

#define EACH_EXPERT_WEIGHTS_OFFSET_SIZE 64
struct mlp_weights_mem {
    memory::ptr weights_base;
    // weights/scale/zp offsets, each expert has 9*4 = 36 bytes
    // gate_weight_offset, up_weight_offset, down_weight_offset
    // gate_scale_offset, up_scale_offset, down_scale_offset
    // gate_zp_offset, up_zp_offset, down_zp_offset
    memory::ptr weights_offset;
};

static void create_weights_memory(mlp_weights_mem& wei_mem, const cldnn::MOE::Config& config, cldnn::engine& engine,
    std::vector<mlp_params>& params) {
    cldnn::memory::ptr weights_base = wei_mem.weights_base;
    size_t weights_offset = 0;
    auto alloc = [&] (ov::Shape shape, ov::element::Type type) {
        auto format = cldnn::format::get_default_format(shape.size());
        cldnn::data_types out_dtype = cldnn::element_type_to_data_type(type);
        auto layout = cldnn::layout(shape, out_dtype, format);
        auto mem = engine.create_subbuffer(*weights_base, layout, weights_offset);
        weights_offset += layout.bytes_count();
        return mem;
    };
    params.resize(config.expert_num);

    for (size_t i = 0; i < config.expert_num; i++) {
        for (size_t j = 0; j < 3; j++) {
            params[i].param[j].weight = alloc({config.hidden_size * config.intermediate_size}, config.weight_type);
            if (config.scale_type != ov::element::dynamic)
                params[i].param[j].scale = alloc(
                    {config.hidden_size * config.intermediate_size / (config.group_size ? config.group_size : 1)}, config.scale_type);
            if (config.zp_type != ov::element::dynamic)
                params[i].param[j].zp = alloc(
                    {config.hidden_size * config.intermediate_size / (config.group_size ? config.group_size : 1)}, config.zp_type);
        }
    }

    cldnn::layout offset_layout(ov::PartialShape{static_cast<int>(config.expert_num), EACH_EXPERT_WEIGHTS_OFFSET_SIZE},
                                cldnn::data_types::i8,
                                cldnn::format::byfx);
    wei_mem.weights_offset = engine.create_subbuffer(*weights_base, offset_layout, weights_offset);
}

/// @brief moe primitive
/// @details Performs moe expert
struct moe : public primitive_base<moe> {
    CLDNN_DECLARE_PRIMITIVE(moe)

    moe() : primitive_base("", {}) {}

    /// @brief Constructs moe primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    moe(const primitive_id& id,
            const std::vector<input_info>& inputs,
            const MOE::Config& config, const std::vector<mlp_params>& param,
            const mlp_weights_mem& wei_mem)
        : primitive_base(id, inputs, 1, {optional_data_type()}),
          _config(config),
          _mlp_params(param),
          _mlp_weights_mem(wei_mem) {
    }

    MOE::Config _config;
    std::vector<mlp_params> _mlp_params;
    mlp_weights_mem _mlp_weights_mem;

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const moe>(rhs);

        return std::memcmp(&_config, &rhs_casted._config, sizeof(_config)) == 0 &&
               _mlp_params == rhs_casted._mlp_params;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe>::save(ob);
        ob << make_data(&_config, sizeof(_config));
        ob << _mlp_weights_mem.weights_base;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe>::load(ib);
        ib >> make_data(&_config, sizeof(_config));
        ib >> _mlp_weights_mem.weights_base;
        create_weights_memory(_mlp_weights_mem, _config, ib.get_engine(), _mlp_params);
    }
};

}  // namespace cldnn
