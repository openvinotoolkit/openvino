// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_nonzero.hpp"

#include "vpu/ngraph/operations/static_shape_nonzero.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

#include "ngraph/graph_util.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticShapeNonZero(std::shared_ptr<ngraph::Node> nonZero) {
    auto staticShapeNonZero = std::make_shared<ngraph::vpu::op::StaticShapeNonZero>(nonZero->input(0).get_source_output());
    staticShapeNonZero->set_friendly_name(nonZero->get_friendly_name() + "/static_shape");

    auto dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
        staticShapeNonZero->output(0), staticShapeNonZero->output(1));
    dynamicShapeResolver->set_friendly_name(nonZero->get_friendly_name() + "/resolve_shape");

    ngraph::replace_node(std::move(nonZero), std::move(dynamicShapeResolver));
}

}  // namespace vpu

