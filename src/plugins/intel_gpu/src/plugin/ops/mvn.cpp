// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/mvn.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/mvn.hpp"

#include <algorithm>

namespace ov {
namespace intel_gpu {

static void CreateCommonMVNOp(Program& p, const std::shared_ptr<ngraph::Node>& op,
                              bool across_channels, bool normalize_variance, float eps, bool eps_inside_sqrt = true) {
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto mvnPrim = cldnn::mvn(layerName,
                              inputPrimitives[0],
                              normalize_variance,
                              eps,
                              eps_inside_sqrt,
                              across_channels,
                              op->get_friendly_name());

    p.AddPrimitive(mvnPrim);
    p.AddPrimitiveToProfiler(op);
}

static void CreateMVNOp(Program& p, const std::shared_ptr<ngraph::op::v0::MVN>& op) {
    p.ValidateInputs(op, {1});

    bool across_channels = op->get_across_channels();
    bool normalize_variance = op->get_normalize_variance();
    float eps = op->get_eps();

    CreateCommonMVNOp(p, op, across_channels, normalize_variance, eps);
}

static void CreateMVNOp(Program& p, const std::shared_ptr<ngraph::op::v6::MVN>& op) {
    p.ValidateInputs(op, {2});

    auto inConst = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
    if (!inConst)
        IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";

    auto& mvnShape = op->get_output_shape(0);
    std::vector<int32_t> axes = inConst->cast_vector<int32_t>();
    for (int32_t& axis : axes)
        axis = axis < 0 ? axis + mvnShape.size() : axis;

    const size_t chanelAxis = 1;
    bool across_channels = std::find(axes.begin(), axes.end(), chanelAxis) != axes.end();
    bool normalize_variance = op->get_normalize_variance();
    float eps = op->get_eps();
    bool eps_inside_sqrt = op->get_eps_mode() == ngraph::op::MVNEpsMode::INSIDE_SQRT;

    CreateCommonMVNOp(p, op, across_channels, normalize_variance, eps, eps_inside_sqrt);
}

REGISTER_FACTORY_IMPL(v0, MVN);
REGISTER_FACTORY_IMPL(v6, MVN);

}  // namespace intel_gpu
}  // namespace ov
