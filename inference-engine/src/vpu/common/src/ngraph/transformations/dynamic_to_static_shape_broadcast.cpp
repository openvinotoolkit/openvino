// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_broadcast.hpp"

#include "vpu/ngraph/operations/static_shape_broadcast.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/utils/error.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticShapeBroadcast(std::shared_ptr<ngraph::Node> target) {
    const auto broadcast = ngraph::as_type_ptr<ngraph::opset3::Broadcast>(target);
    VPU_THROW_UNLESS(broadcast,
                     "dynamicToStaticShapeBroadcast transformation is not applicable for {}, "
                     "it should be {} instead",
                     target, ngraph::opset3::Broadcast::type_info.name);

    std::shared_ptr<ngraph::vpu::op::StaticShapeBroadcast> staticShapeBroadcast;
    if (broadcast->get_broadcast_spec() == ngraph::op::BroadcastType::EXPLICIT) {
        staticShapeBroadcast = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
                broadcast->input_value(0),
                broadcast->input_value(1),
                broadcast->input_value(2));
    } else if (broadcast->get_broadcast_spec() == ngraph::op::BroadcastType::NUMPY) {
        staticShapeBroadcast = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
                broadcast->input_value(0),
                broadcast->input_value(1));
    } else {
        VPU_THROW_FORMAT("dynamicToStaticShapeBroadcast supports only explicit and numpy modes,"
                         "provided {}", broadcast->get_broadcast_spec().m_type);
    }

    auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
            staticShapeBroadcast->output(0), broadcast->input_value(1));

    ngraph::replace_node(std::move(target), std::move(dsr));
}

}  // namespace vpu

