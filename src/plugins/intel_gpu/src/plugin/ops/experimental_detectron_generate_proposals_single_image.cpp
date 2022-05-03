// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/experimental_detectron_generate_proposals.hpp"

#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/experimental_detectron_generate_proposals_single_image.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateExperimentalDetectronGenerateProposalsSingleImageOp(
        Program& p,
        const std::shared_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>& op) {
    p.ValidateInputs(op, {4});
    if (op->get_output_size() != 2) {
        IE_THROW() << "ExperimentalDetectronGenerateProposalsSingleImage requires 2 outputs";
    }

    auto inputs = p.GetInputPrimitiveIDs(op);

    const auto& attrs = op->get_attrs();

    const auto op_friendly_name = op->get_friendly_name();

    const auto layer_type_name = layer_type_name_ID(op);
    const auto layer_name_refine_anchors = layer_type_name + "_refine_anchors";
    const auto layer_name = layer_type_name + ".0";



    const auto mutable_precision = op->get_output_element_type(1);
    const auto output_shape = op->get_output_shape(1);
    const cldnn::layout mutable_layout{DataTypeFromPrecision(mutable_precision),
                                       DefaultFormatForDims(output_shape.size()),
                                       tensor_from_dims(output_shape)};
    cldnn::memory::ptr shared_memory{p.GetEngine().allocate_memory(mutable_layout)};

    const auto mutable_id_w = layer_type_name + "_md_write";
    const cldnn::mutable_data mutable_prim_w{mutable_id_w, shared_memory, op_friendly_name};
    p.primitiveIDs[mutable_id_w] = mutable_id_w;
    p.AddPrimitive(mutable_prim_w);
    inputs.push_back(mutable_id_w);

    const cldnn::experimental_detectron_generate_proposals_single_image prim{layer_name,
                             inputs[0], inputs[1], inputs[2], inputs[3], inputs.back(),
                             attrs.min_size, attrs.nms_threshold, attrs.pre_nms_count, attrs.post_nms_count,
                             op_friendly_name};

    p.AddPrimitive(prim);

    const auto mutable_id_r = layer_type_name + ".1";
    const cldnn::mutable_data mutable_prim_r{mutable_id_r, {layer_name}, shared_memory, op_friendly_name};
    p.primitiveIDs[mutable_id_r] = mutable_id_r;
    p.AddPrimitive(mutable_prim_r);

    p.AddPrimitiveToProfiler(prim, op);
}

REGISTER_FACTORY_IMPL(v6, ExperimentalDetectronGenerateProposalsSingleImage);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
