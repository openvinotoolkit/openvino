// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/vl_sdpa.hpp>
#include "ov_ops/multiclass_nms_ie_internal.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/multiclass_nms.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"

namespace ov::intel_gpu {

static void CreateVLSDPAOp(ProgramBuilder& p, const std::shared_ptr<op::internal::VLSDPA>& op) {
    validate_inputs_count(op, {4});

    auto inputs = p.GetInputInfo(op);

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::vl_sdpa vlsdpa_prim(layerName, inputs,
                                    op->get_input0_transpose_order(),
                                    op->get_input1_transpose_order(),
                                    op->get_input2_transpose_order(),
                                    op->get_output_transpose_order());

    p.add_primitive(*op, vlsdpa_prim);
}

REGISTER_FACTORY_IMPL(internal, VLSDPA);

}  // namespace ov::intel_gpu
