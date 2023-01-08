// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_nonzero.hpp"

#include "vpu/ngraph/operations/static_shape_nonzero.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"
#include "vpu/utils/error.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/ops.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticShapeNonZero(std::shared_ptr<ngraph::Node> node) {
    auto nonZero = std::dynamic_pointer_cast<ngraph::op::v3::NonZero>(node);
    VPU_THROW_UNLESS(nonZero, "dynamicToStaticShapeNonZero transformation for {} of type {} expects {} as node for replacement",
                     node->get_friendly_name(), node->get_type_info(), ngraph::op::v3::NonZero::get_type_info_static());

    auto staticShapeNonZero = std::make_shared<ngraph::vpu::op::StaticShapeNonZero>(nonZero->input(0).get_source_output(), nonZero->get_output_type());

    auto dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
        staticShapeNonZero->output(0), staticShapeNonZero->output(1));
    dynamicShapeResolver->set_friendly_name(nonZero->get_friendly_name());

    ngraph::replace_node(std::move(nonZero), std::move(dynamicShapeResolver));
}

}  // namespace vpu

