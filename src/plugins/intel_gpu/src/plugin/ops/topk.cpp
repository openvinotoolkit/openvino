// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/topk.hpp"

#include "intel_gpu/primitives/arg_max_min.hpp"

namespace ov {
namespace intel_gpu {

static void TopKImpl(ProgramBuilder& p,
                     const std::shared_ptr<ov::Node>& op,
                     ov::op::TopKMode mode,
                     ov::op::TopKSortType stype,
                     uint32_t top_k,
                     uint64_t chosen_axis,
                     bool stable = false) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    size_t num_outputs = op->get_output_size();

    auto topk_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->input_value(1).get_node_shared_ptr());
    auto argmaxPrim = cldnn::arg_max_min(layerName,
                                        inputs[0],
                                        inputs[1],
                                        mode,
                                        (topk_constant ? top_k : 0),
                                        chosen_axis,
                                        stype,
                                        true,
                                        stable,
                                        cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                                        num_outputs);
    argmaxPrim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, argmaxPrim);
}

static void CreateTopKOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::TopK>& op) {
    TopKImpl(p, op, op->get_mode(), op->get_sort_type(), static_cast<uint32_t>(op->get_k()), op->get_axis());
}

static void CreateTopKOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v11::TopK>& op) {
    TopKImpl(p, op, op->get_mode(), op->get_sort_type(), static_cast<uint32_t>(op->get_k()), op->get_axis(), op->get_stable());
}

REGISTER_FACTORY_IMPL(v1, TopK);
REGISTER_FACTORY_IMPL(v11, TopK);

}  // namespace intel_gpu
}  // namespace ov
