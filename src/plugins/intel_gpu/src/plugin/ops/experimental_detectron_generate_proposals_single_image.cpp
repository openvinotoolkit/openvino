// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/experimental_detectron_generate_proposals.hpp"

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

    cldnn::experimental_detectron_generate_proposals_single_image prim{layer_type_name_ID(op),
                             inputs[0], inputs[1], inputs[2], inputs[3],
                             attrs.min_size, attrs.nms_threshold, attrs.pre_nms_count, attrs.post_nms_count};

    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v6, ExperimentalDetectronGenerateProposalsSingleImage);

}  // namespace intel_gpu
}  // namespace ov
