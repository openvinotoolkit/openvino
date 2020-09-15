// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_reshape.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/operations/out_shape_of_reshape.hpp"
#include "vpu/ngraph/utilities.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <memory>
#include <vpu/ngraph/operations/static_shape_reshape.hpp>

namespace vpu {

void dynamicToStaticShapeReshape(std::shared_ptr<ngraph::Node> target) {
    const auto reshape = ngraph::as_type_ptr<ngraph::opset3::Reshape>(target);
    VPU_THROW_UNLESS(reshape, "dynamicToStaticShapeReshape transformation is not applicable for {}, it should be {} instead",
                     target, ngraph::opset3::Reshape::type_info);

    const auto dsr = target->input_value(0).get_node_shared_ptr();
    VPU_THROW_UNLESS(ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(dsr),
                     "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
                     target->get_friendly_name(), target->get_type_info(), ngraph::vpu::op::DynamicShapeResolver::type_info, 0);

    const auto outShapeDescriptor = reshape->input_value(1).get_node_shared_ptr();

    const auto replacement = ngraph::is_type<ngraph::opset3::Constant>(outShapeDescriptor)
        ? reshape->clone_with_new_inputs(reshape->input_values())
        : std::make_shared<ngraph::vpu::op::StaticShapeReshape>(reshape);

    const auto inDataShape = dsr->input(1).get_source_output();
    const auto outShapeOfReshape = std::make_shared<ngraph::vpu::op::OutShapeOfReshape>(inDataShape, outShapeDescriptor, reshape->get_special_zero());

    auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(replacement, outShapeOfReshape);
    outDSR->set_friendly_name(reshape->get_friendly_name());
    ngraph::replace_node(std::move(target), std::move(outDSR));
}

}  // namespace vpu
