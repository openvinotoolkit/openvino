// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_binary_elementwise.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset5.hpp"
#include <ngraph/ops.hpp>

#include <memory>

namespace vpu {

namespace {

void processBinaryEltwise(std::shared_ptr<ngraph::Node> eltwise, size_t lhsIndex, size_t rhsIndex) {
    const auto lhsRank = eltwise->input_value(lhsIndex).get_partial_shape().rank();
    const auto rhsRank = eltwise->input_value(rhsIndex).get_partial_shape().rank();

    const auto copied = eltwise->copy_with_new_inputs(eltwise->input_values());

    const auto lhsDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(eltwise->input_value(lhsIndex).get_node_shared_ptr());
    const auto rhsDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(eltwise->input_value(rhsIndex).get_node_shared_ptr());

    VPU_THROW_UNLESS(lhsDSR || rhsDSR, "DynamicToStaticShape transformation for {} of type {} expects at least one DSR as input",
                     eltwise->get_friendly_name(), eltwise->get_type_info());
    if (lhsDSR && rhsDSR) {
        VPU_THROW_UNLESS(lhsDSR->get_input_element_type(1) == rhsDSR->get_input_element_type(1),
                         "DynamicToStaticShape transformation for {} of type {} expects equal shapes data types, actual {} vs {}",
                         eltwise->get_friendly_name(), eltwise->get_type_info(),
                         lhsDSR->get_input_element_type(1), rhsDSR->get_input_element_type(1));
    }
    const auto shapeElementType = lhsDSR ? lhsDSR->get_input_element_type(1) : rhsDSR->get_input_element_type(1);

    auto lhsInput = lhsDSR ? lhsDSR->input_value(1) : shapeToConstant(shapeElementType, eltwise->get_input_shape(lhsIndex));
    auto rhsInput = rhsDSR ? rhsDSR->input_value(1) : shapeToConstant(shapeElementType, eltwise->get_input_shape(rhsIndex));

    const auto diff = std::abs(lhsRank.get_length() - rhsRank.get_length());
    if (diff) {
        auto & broadcastInput = lhsRank.get_length() < rhsRank.get_length() ? lhsInput : rhsInput;
        const auto broadcastConst = ngraph::opset3::Constant::create(broadcastInput.get_element_type(), {static_cast<size_t>(diff)}, {1});
        broadcastInput = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{broadcastConst, broadcastInput}, 0);
    }

    const auto shape = std::make_shared<ngraph::opset3::Maximum>(lhsInput, rhsInput);

    auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, shape);
    outDSR->set_friendly_name(eltwise->get_friendly_name());
    ngraph::replace_node(std::move(eltwise), std::move(outDSR));
}

} // namespace

void dynamicToStaticShapeBinaryEltwise(std::shared_ptr<ngraph::Node> eltwise) {
    if (eltwise->get_type_info() == ngraph::opset5::Select::type_info) {
        processBinaryEltwise(eltwise, 1, 2);
    } else {
        VPU_THROW_UNLESS(eltwise->get_input_size() == 2,
                         "DynamicToStaticShape transformation for {} of type {} expects two inputs while {} were provided",
                         eltwise->get_friendly_name(), eltwise->get_type_info(), eltwise->get_input_size());
        processBinaryEltwise(eltwise, 0, 1);
    }
}

}  // namespace vpu
