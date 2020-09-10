// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_roialign.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticShapeROIAlign(std::shared_ptr<ngraph::Node> target) {
    const auto roi_align = std::dynamic_pointer_cast<ngraph::opset3::ROIAlign>(target);
    VPU_THROW_UNLESS(roi_align,
        "dynamicToStaticShapeROIAlign transformation is not applicable for {}, it should be {} instead",
        target, ngraph::opset3::ROIAlign::type_info);

    auto shapeToConstant = [&roi_align](const ngraph::Output<ngraph::Node> & output) -> std::shared_ptr<ngraph::opset3::Constant> {
        VPU_THROW_UNLESS(output.get_partial_shape().is_static(),
                         "DynamicToStaticShape transformation for {} of type {} expects static shape on inputs without DSR",
                         roi_align->get_friendly_name(), roi_align->get_type_info());
        return ngraph::opset3::Constant::create(ngraph::element::i64, {output.get_shape().size()}, output.get_shape());
    };

    const auto dataDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(roi_align->input_value(0).get_node_shared_ptr());
    const auto num_roisDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(roi_align->input_value(2).get_node_shared_ptr());

    VPU_THROW_UNLESS(dataDSR || num_roisDSR, "DynamicToStaticShape transformation for {} of type {} expects at least one DSR as input",
                     roi_align->get_friendly_name(), roi_align->get_type_info());

    auto input_0_shape = dataDSR ? dataDSR->input_value(1) : shapeToConstant(roi_align->input_value(0));
    auto num_rois = num_roisDSR ? num_roisDSR->input_value(1) : shapeToConstant(roi_align->input_value(2));

    const auto c_index = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{1});
    const auto c_axis = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{0});
    const auto c = std::make_shared<ngraph::opset3::Gather>(input_0_shape, c_index, c_axis);

    const auto pooled_h = std::make_shared<ngraph::opset3::Constant>(
            input_0_shape.get_element_type(), ngraph::Shape{1}, std::vector<int64_t>{roi_align->get_pooled_h()});
    const auto pooled_w = std::make_shared<ngraph::opset3::Constant>(
            input_0_shape.get_element_type(), ngraph::Shape{1}, std::vector<int64_t>{roi_align->get_pooled_w()});

    const auto output_shape = std::make_shared<ngraph::opset3::Concat>(
            ngraph::OutputVector{num_rois, c, pooled_h, pooled_w}, 0);

    const auto copied = target->clone_with_new_inputs(target->input_values());

    auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, output_shape);
    outDSR->set_friendly_name(roi_align->get_friendly_name());
    ngraph::replace_node(target, std::move(outDSR));
}

}  // namespace vpu
