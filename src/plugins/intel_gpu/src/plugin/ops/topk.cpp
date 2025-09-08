// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/topk.hpp"

#include "intel_gpu/primitives/arg_max_min.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace ov::intel_gpu {

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

    if (p.use_new_shape_infer()) {
        size_t num_outputs = op->get_output_size();

        auto topk_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->input_value(1).get_node_shared_ptr());
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
    } else {
        if (op->get_output_size() == 2) {
            auto mutable_precision = op->get_output_element_type(1);
            if (mutable_precision == ov::element::i64) {
                mutable_precision = ov::element::i32;
            }

            cldnn::layout mutableLayout = cldnn::layout(cldnn::element_type_to_data_type(mutable_precision),
                                                        cldnn::format::get_default_format(op->get_output_shape(1).size()),
                                                        tensor_from_dims(op->get_output_shape(1)));

            GPU_DEBUG_LOG << "[" << layer_type_name_ID(op) << ": mutable data]" << std::endl;
            auto shared_memory = p.get_engine().allocate_memory(mutableLayout);

            cldnn::primitive_id argmax_mutable_id_w = layer_type_name_ID(op) + "_md_write";
            auto argmax_mutable_prim = cldnn::mutable_data(argmax_mutable_id_w,
                                                           shared_memory);
            p.add_primitive(*op, argmax_mutable_prim);
            inputs.push_back(cldnn::input_info(argmax_mutable_id_w));

            std::string ArgMaxLayerName = layerName + ".out0";
            auto argmaxPrim = cldnn::arg_max_min(ArgMaxLayerName,
                                                 inputs,
                                                 mode,
                                                 top_k,
                                                 chosen_axis,
                                                 stype,
                                                 true,
                                                 stable,
                                                 cldnn::element_type_to_data_type(op->get_output_element_type(0)));

            p.add_primitive(*op, argmaxPrim);

            cldnn::primitive_id argmax_mutable_id_r = layerName + ".out1";
            auto argmax_mutable_prim_r = cldnn::mutable_data(argmax_mutable_id_r,
                                                             { cldnn::input_info(ArgMaxLayerName) },
                                                             shared_memory);
            p.add_primitive(*op, argmax_mutable_prim_r);
        } else if (op->get_output_size() == 1) {
            auto argmaxPrim = cldnn::arg_max_min(layerName,
                                                 inputs,
                                                 mode,
                                                 top_k,
                                                 chosen_axis,
                                                 stype,
                                                 true,
                                                 stable,
                                                 cldnn::element_type_to_data_type(op->get_output_element_type(0)));

            p.add_primitive(*op, argmaxPrim);
        } else {
            OPENVINO_THROW(op->get_friendly_name(), " Incorrect TopK outputs number");
        }
    }
}

static void CreateTopKOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::TopK>& op) {
    TopKImpl(p, op, op->get_mode(), op->get_sort_type(), static_cast<uint32_t>(op->get_k()), op->get_axis());
}

static void CreateTopKOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v11::TopK>& op) {
    TopKImpl(p, op, op->get_mode(), op->get_sort_type(), static_cast<uint32_t>(op->get_k()), op->get_axis(), op->get_stable());
}

REGISTER_FACTORY_IMPL(v1, TopK);
REGISTER_FACTORY_IMPL(v11, TopK);

}  // namespace ov::intel_gpu
