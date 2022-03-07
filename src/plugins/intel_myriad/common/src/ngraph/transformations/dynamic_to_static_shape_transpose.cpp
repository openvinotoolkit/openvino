// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_transpose.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticShapeTranspose(std::shared_ptr<ngraph::Node> target) {
    const auto transpose = ngraph::as_type_ptr<ngraph::opset3::Transpose>(target);
    VPU_THROW_UNLESS(transpose, "dynamicToStaticShapeTranspose transformation is not applicable for {}, it should be {} instead",
                     target, ngraph::opset3::Transpose::get_type_info_static());

    const auto dsr = target->input_value(0).get_node_shared_ptr();
    VPU_THROW_UNLESS(ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(dsr),
        "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
        target->get_friendly_name(), target->get_type_info(), ngraph::vpu::op::DynamicShapeResolver::get_type_info_static(), 0);

    const auto transposition = target->input_value(1).get_node_shared_ptr();
    VPU_THROW_UNLESS(ngraph::as_type_ptr<ngraph::opset3::Constant>(transposition),
        "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
        target->get_friendly_name(), target->get_type_info(), ngraph::opset3::Constant::get_type_info_static(), 1);

    const auto copied = transpose->clone_with_new_inputs(target->input_values());
    const auto shape = dsr->input(1).get_source_output();

    const auto axis = std::make_shared<ngraph::opset3::Constant>(
        ngraph::element::i64,
        ngraph::Shape{std::initializer_list<std::size_t>{1}},
        std::vector<std::int64_t>{0});
    const auto gather = std::make_shared<ngraph::opset3::Gather>(shape, transposition, axis);

    auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, gather);
    outDSR->set_friendly_name(transpose->get_friendly_name());
    ngraph::replace_node(std::move(target), std::move(outDSR));
}

}  // namespace vpu
