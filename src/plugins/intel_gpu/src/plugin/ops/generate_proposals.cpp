// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/generate_proposals.hpp"

#include <ov_ops/generate_proposals_ie_internal.hpp>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"

namespace ov {
namespace intel_gpu {

static void CreateGenerateProposalsIEInternalOp(
    ProgramBuilder& p,
    const std::shared_ptr<ov::op::internal::GenerateProposalsIEInternal>& op) {
    validate_inputs_count(op, {4});
    if (op->get_output_size() != 3) {
        OPENVINO_THROW("GenerateProposals requires 3 outputs");
    }

    auto inputs = p.GetInputInfo(op);
    const auto& attrs = op->get_attrs();
    if (p.use_new_shape_infer()) {
        cldnn::generate_proposals prim{layer_type_name_ID(op), inputs, op->get_attrs()};

        prim.num_outputs = op->get_output_size();
        prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});

        p.add_primitive(*op, prim);
    } else {
        const auto layer_type_name = layer_type_name_ID(op);
        const auto layer_name = layer_type_name + ".out0";

        // output 2 - scores
        const auto mutable_precision_1 = op->get_output_element_type(1);
        const auto output_shape_1 = op->get_output_shape(1);
        const cldnn::layout mutable_layout_1{cldnn::element_type_to_data_type(mutable_precision_1),
                                            cldnn::format::get_default_format(output_shape_1.size()),
                                            tensor_from_dims(output_shape_1)};
        cldnn::memory::ptr shared_memory_1{p.get_engine().allocate_memory(mutable_layout_1)};

        const auto mutable_id_w_1 = layer_type_name + "_md_write.1";
        const cldnn::mutable_data mutable_prim_w_1{mutable_id_w_1, shared_memory_1};
        p.add_primitive(*op, mutable_prim_w_1);
        inputs.push_back(cldnn::input_info(mutable_id_w_1));

        // output 3 - roisNum
        const auto output_shape_2 = op->get_output_shape(2);
        const auto mutable_precision_2 = op->get_output_element_type(2);
        const cldnn::layout mutable_layout_2{cldnn::element_type_to_data_type(mutable_precision_2),
                                            cldnn::format::get_default_format(output_shape_2.size()),
                                            tensor_from_dims(output_shape_2)};
        cldnn::memory::ptr shared_memory_2{p.get_engine().allocate_memory(mutable_layout_2)};

        const auto mutable_id_w_2 = layer_type_name + "_md_write.2";
        const cldnn::mutable_data mutable_prim_w_2{mutable_id_w_2, shared_memory_2};
        p.add_primitive(*op, mutable_prim_w_2);
        inputs.push_back(cldnn::input_info(mutable_id_w_2));

        const cldnn::generate_proposals prim{layer_name,
                                             inputs,
                                             attrs.min_size,
                                             attrs.nms_threshold,
                                             attrs.pre_nms_count,
                                             attrs.post_nms_count,
                                             attrs.normalized,
                                             attrs.nms_eta,
                                             cldnn::element_type_to_data_type(op->get_roi_num_type())};

        p.add_primitive(*op, prim);

        const auto mutable_id_r_1 = layer_type_name + ".out1";
        const cldnn::mutable_data mutable_prim_r_1{mutable_id_r_1, {cldnn::input_info(layer_name)}, shared_memory_1};
        p.add_primitive(*op, mutable_prim_r_1);

        const auto mutable_id_r_2 = layer_type_name + ".out2";
        const cldnn::mutable_data mutable_prim_r_2{mutable_id_r_2, {cldnn::input_info(layer_name)}, shared_memory_2};
        p.add_primitive(*op, mutable_prim_r_2);
    }
}

REGISTER_FACTORY_IMPL(internal, GenerateProposalsIEInternal);

}  // namespace intel_gpu
}  // namespace ov
