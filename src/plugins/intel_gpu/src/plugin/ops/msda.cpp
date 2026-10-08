// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/msda.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "ov_ops/msda.hpp"

namespace ov::intel_gpu {

static void CreateMSDAOp(ProgramBuilder& p, const std::shared_ptr<op::internal::MSDA>& op) {
    validate_inputs_count(op, {5});
    auto inputs = p.GetInputInfo(op);

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::msda msda_prim(layerName, inputs);

    p.add_primitive(*op, msda_prim);
}

REGISTER_FACTORY_IMPL(internal, MSDA);

}  // namespace ov::intel_gpu
