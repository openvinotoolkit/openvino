// Copyright (C) 2018-2023 Intel Corporation
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
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto mvnPrim = cldnn::mvn(layerName,
                              inputs[0],
                              normalize_variance,
                              eps,
                              eps_inside_sqrt,
                              across_channels);

    p.add_primitive(*op, mvnPrim);
}

static void CreateMVNOp(Program& p, const std::shared_ptr<ngraph::op::v0::MVN>& op) {
    validate_inputs_count(op, {1});

    bool across_channels = op->get_across_channels();
    bool normalize_variance = op->get_normalize_variance();
    float eps = op->get_eps();

    CreateCommonMVNOp(p, op, across_channels, normalize_variance, eps);
}

static void CreateMVNOp(Program& p, const std::shared_ptr<ngraph::op::v6::MVN>& op) {
    validate_inputs_count(op, {2});

    auto inConst = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
    if (!inConst)
        IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";

    std::vector<int64_t> axes = inConst->cast_vector<int64_t>();
    OPENVINO_SUPPRESS_DEPRECATED_START
    ov::normalize_axes(op.get(), op->get_output_partial_shape(0).size(), axes);
    OPENVINO_SUPPRESS_DEPRECATED_END

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
