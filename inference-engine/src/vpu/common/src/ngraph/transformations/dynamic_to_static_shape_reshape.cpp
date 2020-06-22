// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_reshape.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/operations/out_shape_of_reshape.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticShapeReshape(std::shared_ptr<ngraph::Node> target) {
    const auto dsr = target->get_argument(0);
    VPU_THROW_UNLESS(ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(dsr),
                     "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
                     target->get_friendly_name(), target->get_type_info(), ngraph::vpu::op::DynamicShapeResolver::type_info, 0);

    const auto outShapeDescriptor = target->get_argument(1);
    VPU_THROW_UNLESS(ngraph::as_type_ptr<ngraph::opset3::Constant>(outShapeDescriptor),
                     "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
                     target->get_friendly_name(), target->get_type_info(), ngraph::opset3::Constant::type_info, 1);

    const auto reshape = std::dynamic_pointer_cast<ngraph::opset3::Reshape>(target);
    const auto copied = reshape->clone_with_new_inputs(target->input_values());
    const auto inDataShape = dsr->input(1).get_source_output();

    const auto outShapeOfReshape = std::make_shared<ngraph::vpu::op::OutShapeOfReshape>(
            inDataShape, outShapeDescriptor, reshape->get_special_zero());

    ngraph::replace_node(std::move(target), std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
            copied, outShapeOfReshape));
}

}  // namespace vpu
