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

extern std::string file_path;
extern size_t offload_to_disk;

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

template<typename Type>
void repack(Type* dst, const Type* src, size_t N, size_t K) {
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k++) {
            dst[k * N + n] = src[n * K + k];
        }
    }
}

static bool repack_zp_scale(std::vector<uint8_t>& dst, const uint8_t* src, const ov::Shape& shape, const ov::element::Type type) {
    OPENVINO_ASSERT(shape.size() == 3, "repack_zp_scale expects zp/scale's rank is 3, current is ", shape.size());
    auto groups_count = shape[1];
    if (groups_count == 1)
        return false;
    auto N = shape[0];
    auto K = shape[1];
    auto element_size = ov::shape_size(shape);
    if (type == ov::element::u4 || type == ov::element::i4) {
        dst.resize(element_size / 2);
        for (size_t n = 0; n < N; n += 2) {
            for (size_t k = 0; k < K; k += 2) {
                auto src1 = src[n * K / 2 + k / 2];
                auto src2 = src[(n + 1) * K / 2 + k / 2];
                dst[k * N / 2 + n / 2] = ((src2 & 0xf) << 4) | (src1 & 0xf);
                dst[(k + 1) * N / 2 + n / 2] = (src2 & 0xf0) | ((src1 & 0xf0) >> 4);
            }
        }
    } else if (type == ov::element::u8 || type == ov::element::i8) {
        dst.resize(element_size);
        repack(dst.data(), src, N, K);
    } else if (type == ov::element::f16) {
        dst.resize(element_size * sizeof(uint16_t));
        repack(reinterpret_cast<uint16_t*>(dst.data()), reinterpret_cast<const uint16_t*>(src), N, K);
    } else if (type == ov::element::f32) {
        dst.resize(element_size * sizeof(float));
        repack(reinterpret_cast<float*>(dst.data()), reinterpret_cast<const float*>(src), N, K);
    } else {
        OPENVINO_ASSERT(false, "repack_zp_scale does not support type: ", type);
    }
    return true;
}

static size_t get_weights_size(const std::shared_ptr<ov::op::internal::MOE>& op) {
    size_t weights_size = 0;
    auto get_size = [&](const std::shared_ptr<ov::Node>& node) {
        auto op = ov::as_type_ptr<ov::op::v0::Constant>(node);
        return op->get_byte_size();
    };
    for (size_t i = 0; i < op->get_consts().size(); i++) {
        auto current_consts = op->get_consts()[i];
        for (size_t j = 0; j < 3; j++) {
            weights_size += get_size(current_consts.gates[j]);
            weights_size += get_size(current_consts.ups[j]);
            weights_size += get_size(current_consts.downs[j]);
        }
        // 9*4 = 36 bytes for gate/up/down weight/scale/zp offsets
        // 64-36 = 28 bytes for padding
        weights_size += EACH_EXPERT_WEIGHTS_OFFSET_SIZE;
    }

    return weights_size;
}

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
            const mlp_weights_mem& wei_mem,
            const std::shared_ptr<ov::op::internal::MOE>& op)
        : primitive_base(id, inputs, 1, {optional_data_type()}),
          _config(config),
          _mlp_params(param),
          _mlp_weights_mem(wei_mem),
          _op(op) {
    }

    MOE::Config _config;
    std::vector<mlp_params> _mlp_params;
    mlp_weights_mem _mlp_weights_mem;
    std::vector<size_t> _weights_info;
    std::shared_ptr<ov::op::internal::MOE> _op;

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
