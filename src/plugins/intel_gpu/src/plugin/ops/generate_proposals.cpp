// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/generate_proposals.hpp"

#include <ov_ops/generate_proposals_ie_internal.hpp>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

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
    cldnn::generate_proposals prim{layer_type_name_ID(op), inputs, op->get_attrs()};

    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, GenerateProposalsIEInternal);

}  // namespace intel_gpu
}  // namespace ov
