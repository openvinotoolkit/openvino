// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_transpose.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticShapeTranspose(std::shared_ptr<ngraph::Node> target) {
    const auto dsr = target->get_argument(0);
    VPU_THROW_UNLESS(ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(dsr),
        "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
        target->get_friendly_name(), target->get_type_info(), ngraph::vpu::op::DynamicShapeResolver::type_info, 0);

    const auto transposition = target->get_argument(1);
    VPU_THROW_UNLESS(ngraph::as_type_ptr<ngraph::opset3::Constant>(transposition),
        "DynamicToStaticShape transformation for {] of type {} expects {} as input with index {}",
        target->get_friendly_name(), target->get_type_info(), ngraph::opset3::Constant::type_info, 1);

    const auto transpose = std::dynamic_pointer_cast<ngraph::opset3::Transpose>(target);
    const auto copied = transpose->copy_with_new_args(target->get_arguments());
    const auto shape = dsr->input(1).get_source_output();

    const auto axis = std::make_shared<ngraph::opset3::Constant>(
        ngraph::element::i64,
        ngraph::Shape{std::initializer_list<std::size_t>{1}},
        std::vector<std::int64_t>{0});
    const auto scatterElementsUpdate = std::make_shared<ngraph::opset3::ScatterElementsUpdate>(shape, transposition, shape, axis);
    ngraph::replace_node(std::move(target), std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, scatterElementsUpdate));
}

}  // namespace vpu
