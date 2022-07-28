// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/scatter_update.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/scatter_update.hpp"

namespace ov {
namespace intel_gpu {

static void CreateScatterUpdateOp(Program& p, const std::shared_ptr<ngraph::op::v3::ScatterUpdate>& op) {
    p.ValidateInputs(op, {4});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto axes_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(3));
    if (!axes_constant) {
        IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }
    int64_t axis = axes_constant->cast_vector<int64_t>()[0];
    auto primitive = cldnn::scatter_update(layerName,
                                           inputPrimitives[0],
                                           inputPrimitives[1],
                                           inputPrimitives[2],
                                           axis,
                                           op->get_friendly_name());

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v3, ScatterUpdate);

}  // namespace intel_gpu
}  // namespace ov
