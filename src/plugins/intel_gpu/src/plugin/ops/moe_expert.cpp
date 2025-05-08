// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/core/graph_util.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "ov_ops/moe_expert.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/moe_expert.hpp"

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
    auto element_size = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>());
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

static size_t get_weights_size(const std::shared_ptr<ov::op::internal::MOEExpert>& op, int cm_mask) {
    size_t weights_size = 0;
    auto get_size = [&](const std::shared_ptr<ov::Node>& node) {
        auto op = ov::as_type_ptr<ov::op::v0::Constant>(node);
        ov::Shape const_shape = op->get_shape();
        auto constFormat = cldnn::format::get_default_format(const_shape.size());
        cldnn::data_types out_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));
        auto layout = cldnn::layout(const_shape, out_dtype, constFormat);
        return layout.bytes_count();
    };
    for (size_t i = 0; i < op->get_consts().size(); i++) {
        auto current_consts = op->get_consts()[i];
        for (size_t j = 0; j < 3; j++) {
            weights_size += get_size(current_consts.gate[j]);
            weights_size += get_size(current_consts.up[j]);
            weights_size += get_size(current_consts.down[j]);
            if (cm_mask && j > 0) {
                weights_size += get_size(current_consts.gate[j]);
                weights_size += get_size(current_consts.up[j]);
                weights_size += get_size(current_consts.down[j]);
            }
        }
    }
    if (cm_mask) {
        // [64bytes]->gate_addrs,up_addrs, gate_scales_addrs, up_scales_addrs,gate_zp_addrs,up_zp_addrs, padding1, padding2
        cldnn::layout ptr_layout(ov::PartialShape{static_cast<int>(op->get_config().expert_num)}, cldnn::data_types::u64, cldnn::format::byfx);
        cldnn::layout gate_up_ptr_layout(ov::PartialShape{static_cast<int>(op->get_config().expert_num * 64 / sizeof(uint64_t))},
                                         cldnn::data_types::u64,
                                         cldnn::format::byfx);
        weights_size += gate_up_ptr_layout.bytes_count();
        weights_size += 3 * ptr_layout.bytes_count();
    }
    return weights_size;
}

static cldnn::memory::ptr pre_allocate_weights(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOEExpert>& op, int cm_mask) {
    auto size = get_weights_size(op, cm_mask);
    auto layout = cldnn::layout({1, 1, 1, static_cast<ov::Dimension::value_type>(size)}, ov::element::i8, cldnn::format::bfyx);
    auto alloc_type = p.get_engine().get_preferred_memory_allocation_type(false);
    auto mem = p.get_engine().allocate_memory(layout, alloc_type, false);

    return mem;
}

static void prepare_weights(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOEExpert>& op, std::vector<cldnn::moe_expert::mlp_params>& params,
    cldnn::moe_expert::scale_zp_mems& scale_zp, cldnn::memory::ptr& weights_base, int cm_mask) {

    size_t weights_offset = 0;
    const auto& consts = op->get_consts();
    auto alloc = [&] (const std::shared_ptr<ov::Node>& node, bool try_repack = false) {
        auto op = ov::as_type_ptr<ov::op::v0::Constant>(node);
        ov::Shape const_shape = op->get_shape();
        auto constFormat = cldnn::format::get_default_format(const_shape.size());
        cldnn::data_types out_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));
        auto layout = cldnn::layout(const_shape, out_dtype, constFormat);
        auto data = op->get_data_ptr<uint8_t>();
        const auto cache_key = std::make_tuple(data, const_shape, out_dtype);
        std::vector<uint8_t> repacked_buf;
        if (try_repack) {
            auto repacked = repack_zp_scale(repacked_buf, data, const_shape, op->get_output_element_type(0));
            if (repacked) {
                data = repacked_buf.data();
                auto new_shape = ov::Shape{const_shape[1], const_shape[0], 1};
                layout = cldnn::layout(new_shape, out_dtype, constFormat);
            }
        }

        auto mem = p.get_engine().create_subbuffer(*weights_base, layout, weights_offset);
        weights_offset += layout.bytes_count();

        auto& stream = p.get_engine().get_service_stream();
        mem->copy_from(stream, data, 0, 0, layout.bytes_count(), true);
        return mem;
    };
    OPENVINO_ASSERT(op->get_config().expert_num == consts.size());
    params.resize(consts.size());

    cldnn::layout ptr_layout(ov::PartialShape{static_cast<int>(op->get_config().expert_num)}, cldnn::data_types::u64, cldnn::format::byfx);
    cldnn::layout gate_up_ptr_layout(ov::PartialShape{static_cast<int>(op->get_config().expert_num * 64 / sizeof(uint64_t))},
                                     cldnn::data_types::u64,
                                     cldnn::format::byfx);

    if (cm_mask) {
        // [64bytes]->gate_addrs,up_addrs, gate_scales_addrs, up_scales_addrs,gate_zp_addrs,up_zp_addrs, padding1, padding2
        scale_zp.gate_up_addrs = p.get_engine().create_subbuffer(*weights_base, gate_up_ptr_layout, weights_offset);
        weights_offset += gate_up_ptr_layout.bytes_count();
        scale_zp.down_addrs = p.get_engine().create_subbuffer(*weights_base, ptr_layout, weights_offset);
        weights_offset += ptr_layout.bytes_count();
        scale_zp.down_scales_addrs = p.get_engine().create_subbuffer(*weights_base, ptr_layout, weights_offset);
        weights_offset += ptr_layout.bytes_count();
        scale_zp.down_zp_addrs = p.get_engine().create_subbuffer(*weights_base, ptr_layout, weights_offset);
        weights_offset += ptr_layout.bytes_count();
    }
    std::array<std::vector<uint64_t>, 3> buf_down;
    struct addrs {
        uint64_t gate_addrs;
        uint64_t up_addrs;
        uint64_t gate_scales_addrs;
        uint64_t up_scales_addrs;
        uint64_t gate_zp_addrs;
        uint64_t up_zp_addrs;
        uint64_t padding[2];
    };
    std::vector<addrs> buf_gate_up(op->get_config().expert_num);
    for (size_t i = 0; i < consts.size(); i++) {
        auto current_consts = consts[i];
        params[i].base_addr = weights_base;
#define SET_BUF(src_name, dst_idx) \
        params[i].param[dst_idx].weight = alloc(current_consts.src_name[0]);  \
        buf_gate_up[i].gate_addrs = reinterpret_cast<uint64_t>(params[i].param[dst_idx].weight->buffer_ptr());    \
        params[i].param[dst_idx].scale = alloc(current_consts.src_name[1], true);    \
        params[i].param[dst_idx].zp = alloc(current_consts.src_name[2], true);   \
        if (cm_mask) {  \
            params[i].param[dst_idx].scale_ba = alloc(current_consts.src_name[1], false);    \
            buf_gate_up[i].src_name##_scales_addrs = reinterpret_cast<uint64_t>(params[i].param[dst_idx].scale_ba->buffer_ptr());   \
            params[i].param[dst_idx].zp_ba = alloc(current_consts.src_name[2], false);   \
            buf_gate_up[i].src_name##_zp_addrs = reinterpret_cast<uint64_t>(params[i].param[dst_idx].zp_ba->buffer_ptr());  \
        }

        SET_BUF(gate, 0)
        SET_BUF(up, 1)
#undef SET_BUF

        params[i].param[2].weight = alloc(current_consts.down[0]);
        buf_down[0].push_back(reinterpret_cast<uint64_t>(params[i].param[2].weight->buffer_ptr()));
        params[i].param[2].scale = alloc(current_consts.down[1], true);
        params[i].param[2].zp = alloc(current_consts.down[2], true);
        if (cm_mask) {
            params[i].param[2].scale_ba = alloc(current_consts.down[1], false);
            buf_down[1].push_back(reinterpret_cast<uint64_t>(params[i].param[2].scale_ba->buffer_ptr()));
            params[i].param[2].zp_ba = alloc(current_consts.down[2], false);
            buf_down[2].push_back(reinterpret_cast<uint64_t>(params[i].param[2].zp_ba->buffer_ptr()));
        }
    }

    if (cm_mask) {
        auto& stream = p.get_engine().get_service_stream();
        scale_zp.gate_up_addrs->copy_from(stream, buf_gate_up.data(), 0, 0, gate_up_ptr_layout.bytes_count(), true);
        scale_zp.down_addrs->copy_from(stream, buf_down[0].data(), 0, 0, ptr_layout.bytes_count(), true);
        scale_zp.down_scales_addrs->copy_from(stream, buf_down[1].data(), 0, 0, ptr_layout.bytes_count(), true);
        scale_zp.down_zp_addrs->copy_from(stream, buf_down[2].data(), 0, 0, ptr_layout.bytes_count(), true);
    }

    // std::cout << "weights offset: " << weights_offset << ", weights_size: " << weights_base->size() << std::endl;
}

static void CreateMOEExpertOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOEExpert>& op) {
    auto inputs = p.GetInputInfo(op);
    const auto& config = op->get_config();
    OPENVINO_ASSERT(config.fused_router_logic, "MOEExpert must fuse router logic");
    OPENVINO_ASSERT(inputs.size() == 2, "Inputs count of MOEExpert should be 2");

    const std::string layerName = layer_type_name_ID(op);
    std::vector<cldnn::moe_expert::mlp_params> params;
    cldnn::moe_expert::scale_zp_mems scale_zps;

    int cm_mask = 1;
    auto env = std::getenv("CM_MASK");
    if (env) {
        cm_mask = std::atoi(env);
    }
    auto mem = pre_allocate_weights(p, op, cm_mask);
    prepare_weights(p, op, params, scale_zps, mem, cm_mask);

    const cldnn::moe_expert moe(layerName, inputs, config, params, scale_zps);

    p.add_primitive(*op, moe);
}

REGISTER_FACTORY_IMPL(internal, MOEExpert);

}  // namespace ov::intel_gpu
