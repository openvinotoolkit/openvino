// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/lrn.hpp"
#include "ngraph/op/constant.hpp"

#include "api/lrn.hpp"

namespace CLDNNPlugin {

static cldnn::lrn_norm_region GetNormRegion(std::vector<int64_t> axis_value) {
    if (axis_value.size() == 1 && axis_value[0] == 1) {
        return cldnn::lrn_norm_region_across_channel;
    } else {
        return cldnn::lrn_norm_region_within_channel;
    }
}

void CreateLRNOp(Program& p, const std::shared_ptr<ngraph::op::v0::LRN>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto axis_const = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    if (!axis_const) {
        THROW_IE_EXCEPTION << "Unsupported axes node type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }
    auto axis_value = axis_const->cast_vector<int64_t>();
    auto localSize = op->get_nsize();

    auto lrnPrim = cldnn::lrn(layerName,
                              inputPrimitives[0],
                              localSize,
                              static_cast<float>(op->get_bias()),
                              static_cast<float>(op->get_alpha()),
                              static_cast<float>(op->get_beta()),
                              GetNormRegion(axis_value));

    p.AddPrimitive(lrnPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, LRN);

}  // namespace CLDNNPlugin
