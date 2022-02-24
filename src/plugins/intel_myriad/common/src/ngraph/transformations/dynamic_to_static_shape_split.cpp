// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_split.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"
#include "vpu/utils/error.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/validation_util.hpp"

namespace vpu {

void validateSplit(const ngraph::Node& split) {
    VPU_THROW_UNLESS(split.get_input_size() >= 2, "There is Split operation \"{}\" without specified axis", split.get_friendly_name());
    const auto& axis = ngraph::as_type_ptr<ngraph::opset5::Constant>(split.input_value(1).get_node_shared_ptr());
    VPU_THROW_UNLESS(axis != nullptr, "There is Split operation \"{}\" with dynamic axis \"{}\", but only constant axis is supported",
        split.get_friendly_name(), split.input_value(1).get_node_shared_ptr()->get_friendly_name());
    const auto axisValue = ngraph::normalize_axis(split.description(), axis->cast_vector<std::int64_t>().front(), split.get_input_partial_shape(0).rank());
    VPU_THROW_UNLESS(split.get_input_partial_shape(0)[axisValue].is_static(),
        "There is Split operation \"{}\" by dynamic dimension, but only split by static dimension is supported: shape = \"{}\", axis = \"{}\"",
        split.get_friendly_name(), split.get_input_partial_shape(0), axisValue);
}

void dynamicToStaticShapeSplit(std::shared_ptr<ngraph::Node> target) {
    const auto split = ngraph::as_type_ptr<ngraph::opset5::Split>(target);
    VPU_THROW_UNLESS(split,
                     "dynamicToStaticShapeSplit transformation is not applicable for {}, "
                     "it should be {} instead",
                     target, ngraph::opset5::Split::get_type_info_static());

    const auto numSplits = split->get_num_splits();

    const auto dsr = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(target->input_value(0).get_node_shared_ptr());
    VPU_THROW_UNLESS(dsr, "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
                     target->get_friendly_name(), target->get_type_info(), ngraph::vpu::op::DynamicShapeResolver::get_type_info_static(), 0);

    const auto dataShape = dsr->input_value(1).get_node_shared_ptr();
    const auto dataShapeType = dataShape->get_element_type();
    const auto axisNode = ngraph::as_type_ptr<ngraph::opset5::Constant>(target->input_value(1).get_node_shared_ptr());
    VPU_THROW_UNLESS(axisNode, "dynamicToStaticShapeSplit transformation is not applicable for {}, dynamic axis is not supported", target);

    const auto dataRank = target->get_input_partial_shape(0).rank();
    VPU_THROW_UNLESS(dataRank.is_static(), "dynamicToStaticShapeSplit transformation for {} doesn't support dynamic rank", target);
    const auto srcAxis = axisNode->cast_vector<int64_t>();
    VPU_THROW_UNLESS(srcAxis.size() == 1,
                     "dynamicToStaticShapeSplit transformation for {} failed: axis node is represented as {} values while 1 is expected",
                     target, srcAxis.size());
    const auto resultAxis = ngraph::normalize_axis(target->description(), axisNode->cast_vector<int64_t>()[0], dataRank);

    const auto axisVector = ngraph::opset5::Constant::create(dataShapeType, {1}, {resultAxis});

    const auto dimToSplitBy = std::make_shared<ngraph::opset5::Gather>(dataShape,
                                                                       axisVector,
                                                                       ngraph::opset5::Constant::create(dataShapeType, {1}, {0}));
    const auto splittedDim = std::make_shared<ngraph::opset5::Divide>(dimToSplitBy,
                                                                      ngraph::opset5::Constant::create(dataShapeType, {1}, {numSplits}));
    const auto resultShape = std::make_shared<ngraph::opset5::ScatterElementsUpdate>(dataShape,
                                                                                     axisVector,
                                                                                     splittedDim,
                                                                                     ngraph::opset5::Constant::create(dataShapeType, {1}, {0}));

    const auto copied = target->clone_with_new_inputs(target->input_values());

    for (size_t i = 0; i < copied->get_output_size(); i++) {
        const auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied->output(i), resultShape);
        outDSR->set_friendly_name(target->get_friendly_name() + "." + std::to_string(i));
        target->output(i).replace(outDSR);
    }
}

}  // namespace vpu
