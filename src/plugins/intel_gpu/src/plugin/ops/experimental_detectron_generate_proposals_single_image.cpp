// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/experimental_detectron_generate_proposals.hpp"

#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/experimental_detectron_generate_proposals_single_image.hpp"

namespace ov {
namespace intel_gpu {

static void CreateExperimentalDetectronGenerateProposalsSingleImageOp(
        ProgramBuilder& p,
        const std::shared_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>& op) {
    validate_inputs_count(op, {4});
    if (op->get_output_size() != 2) {
        OPENVINO_THROW("ExperimentalDetectronGenerateProposalsSingleImage requires 2 outputs");
    }

    auto inputs = p.GetInputInfo(op);

    const auto& attrs = op->get_attrs();

    if (p.use_new_shape_infer()) {
        cldnn::experimental_detectron_generate_proposals_single_image prim{layer_type_name_ID(op),
                             inputs[0], inputs[1], inputs[2], inputs[3],
                             attrs.min_size, attrs.nms_threshold, attrs.pre_nms_count, attrs.post_nms_count};

        prim.num_outputs = op->get_output_size();
        prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});

        p.add_primitive(*op, prim);
    } else {
        const auto layer_type_name = layer_type_name_ID(op);
        const auto layer_name = layer_type_name + ".out0";

        const auto mutable_precision = op->get_output_element_type(1);
        const auto output_shape = op->get_output_shape(1);
        const cldnn::layout mutable_layout{cldnn::element_type_to_data_type(mutable_precision),
                                        cldnn::format::get_default_format(output_shape.size()),
                                        tensor_from_dims(output_shape)};
        cldnn::memory::ptr shared_memory{p.get_engine().allocate_memory(mutable_layout)};

        const auto mutable_id_w = layer_type_name + "_md_write";
        const cldnn::mutable_data mutable_prim_w{mutable_id_w, shared_memory};
        p.add_primitive(*op, mutable_prim_w);
        inputs.push_back(cldnn::input_info(mutable_id_w));

        const cldnn::experimental_detectron_generate_proposals_single_image prim{layer_name,
                                inputs[0], inputs[1], inputs[2], inputs[3], inputs.back(),
                                attrs.min_size, attrs.nms_threshold, attrs.pre_nms_count, attrs.post_nms_count};

        p.add_primitive(*op, prim);

        const auto mutable_id_r = layer_type_name + ".out1";
        const cldnn::mutable_data mutable_prim_r{mutable_id_r, {cldnn::input_info(layer_name)}, shared_memory};
        p.add_primitive(*op, mutable_prim_r);
    }
}

REGISTER_FACTORY_IMPL(v6, ExperimentalDetectronGenerateProposalsSingleImage);

}  // namespace intel_gpu
}  // namespace ov
