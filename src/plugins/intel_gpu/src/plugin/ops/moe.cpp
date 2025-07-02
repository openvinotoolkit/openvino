// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/core/graph_util.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "ov_ops/moe.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/moe.hpp"

namespace ov::intel_gpu {

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

static cldnn::memory::ptr pre_allocate_weights(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOE>& op) {
    auto size = get_weights_size(op);
    auto layout = cldnn::layout({1, 1, 1, static_cast<ov::Dimension::value_type>(size)}, ov::element::i8, cldnn::format::bfyx);
    auto alloc_type = p.get_engine().get_preferred_memory_allocation_type(false);
    auto mem = p.get_engine().allocate_memory(layout, alloc_type, false);

    return mem;
}

static void fill_weights_memory(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOE>& op, std::vector<cldnn::mlp_params>& params,
    cldnn::mlp_weights_mem& wei_mem) {
    auto& stream = p.get_engine().get_service_stream();
    auto fill = [&] (const std::shared_ptr<ov::op::v0::Constant>& op, cldnn::memory_ptr mem, bool try_repack = false) {
        if (!mem)
            return;
        ov::Shape const_shape = op->get_shape();
        auto constFormat = cldnn::format::get_default_format(const_shape.size());
        cldnn::data_types out_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));
        auto layout = cldnn::layout(const_shape, out_dtype, constFormat);
        auto data = op->get_data_ptr<uint8_t>();
        std::vector<uint8_t> repacked_buf;
        if (try_repack) {
            auto repacked = repack_zp_scale(repacked_buf, data, const_shape, op->get_output_element_type(0));
            if (repacked) {
                data = repacked_buf.data();
                auto new_shape = ov::Shape{const_shape[1], const_shape[0], 1};
                layout = cldnn::layout(new_shape, out_dtype, constFormat);
            }
        }

        mem->copy_from(stream, data, 0, 0, layout.bytes_count(), true);
    };
    const auto& consts = op->get_consts();
    OPENVINO_ASSERT(op->get_config().expert_num == consts.size());
    params.resize(consts.size());

    auto weights_base_ptr = reinterpret_cast<uint8_t*>(wei_mem.weights_base->buffer_ptr());
    std::vector<uint32_t> offsets;
    offsets.reserve(consts.size() * EACH_EXPERT_WEIGHTS_OFFSET_SIZE / sizeof(uint32_t));
    for (size_t i = 0; i < consts.size(); i++) {
        auto current_consts = consts[i];
#define SET_BUF(src_name, dst_idx)                                                                                           \
        fill(current_consts.src_name[0], params[i].param[dst_idx].weight);                                                   \
        offsets.push_back(reinterpret_cast<uint8_t*>(params[i].param[dst_idx].weight->buffer_ptr()) - weights_base_ptr);     \
        fill(current_consts.src_name[1], params[i].param[dst_idx].scale, true);                                              \
        offsets.push_back(reinterpret_cast<uint8_t*>(params[i].param[dst_idx].scale->buffer_ptr()) - weights_base_ptr);      \
        fill(current_consts.src_name[2], params[i].param[dst_idx].zp, true);                                                 \
        offsets.push_back(reinterpret_cast<uint8_t*>(params[i].param[dst_idx].zp->buffer_ptr()) - weights_base_ptr);

        SET_BUF(gates, 0)
        SET_BUF(ups, 1)
        SET_BUF(downs, 2)
#undef SET_BUF
        // padding
        for (size_t j = 0; j < EACH_EXPERT_WEIGHTS_OFFSET_SIZE / sizeof(uint32_t) - 9; j++) {
            offsets.push_back(0);
        }
    }

    wei_mem.weights_offset->copy_from(stream, offsets.data(), 0, 0, wei_mem.weights_offset->get_layout().bytes_count(), true);
}

static void CreateMOEOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOE>& op) {
    auto inputs = p.GetInputInfo(op);
    const auto& config = op->get_config();
    OPENVINO_ASSERT(inputs.size() == 2, "Inputs count of MOE should be 2");

    const std::string layerName = layer_type_name_ID(op);
    std::vector<cldnn::mlp_params> params;
    cldnn::mlp_weights_mem wei_mem;
    auto& engine = p.get_engine();

    wei_mem.weights_base = pre_allocate_weights(p, op);
    create_weights_memory(wei_mem, config, engine, params);
    fill_weights_memory(p, op, params, wei_mem);

    const cldnn::moe moe(layerName, inputs, config, params, wei_mem);

    p.add_primitive(*op, moe);
}

REGISTER_FACTORY_IMPL(internal, MOE);

}  // namespace ov::intel_gpu
