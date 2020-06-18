// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_binary_elementwise.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticShapeBinaryEltwise(std::shared_ptr<ngraph::Node> eltwise) {
    const auto lhsRank = eltwise->input_value(0).get_partial_shape().rank();
    const auto rhsRank = eltwise->input_value(1).get_partial_shape().rank();

    const auto copied = eltwise->copy_with_new_inputs(eltwise->input_values());

    auto shapeToConstant = [&eltwise](const ngraph::Output<ngraph::Node> & output) -> std::shared_ptr<ngraph::opset3::Constant> {
        VPU_THROW_UNLESS(output.get_partial_shape().is_static(),
            "DynamicToStaticShape transformation for {} of type {} expects static shape on inputs without DSR",
            eltwise->get_friendly_name(), eltwise->get_type_info());
        return ngraph::opset3::Constant::create(ngraph::element::i64, {output.get_shape().size()}, output.get_shape());
    };

    const auto lhsDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(eltwise->input_value(0).get_node_shared_ptr());
    const auto rhsDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(eltwise->input_value(1).get_node_shared_ptr());

    VPU_THROW_UNLESS(lhsDSR || rhsDSR, "DynamicToStaticShape transformation for {} of type {} expects at least one DSR as input",
        eltwise->get_friendly_name(), eltwise->get_type_info());

    auto lhsInput = lhsDSR ? lhsDSR->input_value(1) : shapeToConstant(eltwise->input_value(0));
    auto rhsInput = rhsDSR ? rhsDSR->input_value(1) : shapeToConstant(eltwise->input_value(1));

    const auto diff = std::abs(lhsRank.get_length() - rhsRank.get_length());
    if (diff) {
        auto & broadcastInput = lhsRank.get_length() < rhsRank.get_length() ? lhsInput : rhsInput;
        const auto broadcastConst = ngraph::opset3::Constant::create(broadcastInput.get_element_type(), {static_cast<size_t>(diff)}, {1});
        broadcastInput = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{broadcastConst, broadcastInput}, 0);
    }

    const auto shape = std::make_shared<ngraph::opset3::Maximum>(lhsInput, rhsInput);

    const auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, shape);
    outDSR->set_friendly_name(eltwise->get_friendly_name());
    ngraph::replace_node(std::move(eltwise), outDSR);
}

}  // namespace vpu
