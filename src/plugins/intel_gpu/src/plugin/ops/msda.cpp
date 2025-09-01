// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/msda.hpp>
#include "ov_ops/multiclass_nms_ie_internal.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/multiclass_nms.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"

namespace ov::intel_gpu {

static void CreateMSDAOp(ProgramBuilder& p, const std::shared_ptr<op::internal::MSDA>& op) {
    auto inputs = p.GetInputInfo(op);

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::msda msda_prim(layerName, inputs);

    p.add_primitive(*op, msda_prim);
}

REGISTER_FACTORY_IMPL(internal, MSDA);

}  // namespace ov::intel_gpu