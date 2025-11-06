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

namespace cldnn {
    std::string file_path;
    size_t offload_to_disk = 0;
}

namespace ov::intel_gpu {
static cldnn::memory::ptr pre_allocate_weights(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOE>& op) {
    auto size = cldnn::get_weights_size(op);
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
            auto repacked = cldnn::repack_zp_scale(repacked_buf, data, const_shape, op->get_output_element_type(0));
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

    const auto& model = p.get_model();
    if (model->has_rt_info("offload_to_disk")) {
        cldnn::offload_to_disk = model->get_rt_info()["offload_to_disk"].as<size_t>();
        cldnn::file_path = model->get_rt_info()["__weights_path"].as<std::string>();
    }

    wei_mem.weights_base = pre_allocate_weights(p, op);
    create_weights_memory(wei_mem, config, engine, params);
    if (!cldnn::offload_to_disk) {
        fill_weights_memory(p, op, params, wei_mem);
    }

    const cldnn::moe moe(layerName, inputs, config, params, wei_mem, op);
    p.add_primitive(*op, moe);
}

REGISTER_FACTORY_IMPL(internal, MOE);

}  // namespace ov::intel_gpu
