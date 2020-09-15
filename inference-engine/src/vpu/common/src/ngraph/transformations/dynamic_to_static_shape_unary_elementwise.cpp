// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_unary_elementwise.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/ops.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticUnaryElementwise(std::shared_ptr<ngraph::Node> target) {
    const auto dsr = target->input_value(0).get_node_shared_ptr();
    VPU_THROW_UNLESS(ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(dsr),
                     "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
                     target->get_friendly_name(), target->get_type_info(), ngraph::vpu::op::DynamicShapeResolver::type_info, 0);

    const auto shape = dsr->input(1).get_source_output();
    const auto copied = target->clone_with_new_inputs(target->input_values());

    auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, shape);
    outDSR->set_friendly_name(target->get_friendly_name());
    ngraph::replace_node(target, std::move(outDSR));
}

}  // namespace vpu
