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

static void prepare_weights(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOEExpert2>& op, cldnn::moe_expert::mlp_params& params) {
    auto internal_body = op->get_body();
    auto alloc = [&] (const std::shared_ptr<ov::Node>& node) {
        auto op = ov::as_type_ptr<ov::op::v0::Constant>(node);
        ov::Shape const_shape = op->get_shape();
        auto constFormat = cldnn::format::get_default_format(const_shape.size());
        cldnn::data_types out_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));
        auto layout = cldnn::layout(const_shape, out_dtype, constFormat);
        auto mem = p.get_engine().allocate_memory(layout, false);
        auto data = op->get_data_ptr<char>();
        const auto cache_key = std::make_tuple(data, const_shape, out_dtype);
        auto& stream = p.get_engine().get_service_stream();
        cldnn::mem_lock<char> lock{mem, stream};
        auto buf = lock.data();
        std::memcpy(&buf[0], &data[0], layout.bytes_count());
        return mem;
    };
    for (auto& node : internal_body->get_ordered_ops()) {
        auto& rt = node->get_rt_info();
        if (rt.count("__weight_const__")) {
            auto idx = rt["__weight_const__"].as<int>();
            OPENVINO_ASSERT(idx >= 0 && idx < 3);
            params.param[idx].weight = alloc(node);
        }
        if (rt.count("__scale_const__")) {
            auto idx = rt["__scale_const__"].as<int>();
            OPENVINO_ASSERT(idx >= 0 && idx < 3);
            params.param[idx].scale = alloc(node);
        }
        if (rt.count("__zp_const__")) {
            auto idx = rt["__zp_const__"].as<int>();
            OPENVINO_ASSERT(idx >= 0 && idx < 3);
            params.param[idx].zp = alloc(node);
        }
    }
}

static void CreateMOEExpert2Op(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOEExpert2>& op) {
    auto inputs = p.GetInputInfo(op);
    OPENVINO_ASSERT(inputs.size() == 4, "Inputs count should be 4");

    const std::string layerName = layer_type_name_ID(op);
    cldnn::moe_expert::mlp_params params;
    prepare_weights(p, op, params);

    const cldnn::moe_expert moe(layerName,
                                inputs,
                                op->get_config(),
                                params
                                );

    p.add_primitive(*op, moe);
}

REGISTER_FACTORY_IMPL(internal, MOEExpert2);

}  // namespace ov::intel_gpu
