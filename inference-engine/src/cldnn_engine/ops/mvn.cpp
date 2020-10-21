// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/mvn.hpp"

#include "api/mvn.hpp"

namespace CLDNNPlugin {

void CreateMVNOp(Program& p, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::MVN>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    const size_t chanelAxis = 1;
    ngraph::AxisSet reductionAxes = op->get_reduction_axes();
    // FIXME: op->get_across_channels(); doesn't work for some reason. Is it expected?
    bool across_channels = reductionAxes.count(chanelAxis) > 0;
    bool normalize_variance = op->get_normalize_variance();
    float eps = op->get_eps();

    auto mvnPrim = cldnn::mvn(layerName,
                              inputPrimitives[0],
                              across_channels,
                              normalize_variance,
                              eps);

    p.AddPrimitive(mvnPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, MVN);

}  // namespace CLDNNPlugin
