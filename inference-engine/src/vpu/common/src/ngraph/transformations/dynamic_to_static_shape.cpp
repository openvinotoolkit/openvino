// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape.hpp"

#include "vpu/ngraph/transformations/dynamic_to_static_shape_nonzero.hpp"

#include <vpu/utils/error.hpp>

namespace ngraph {
namespace pass {

bool DynamicToStaticShape::run_on_function(std::shared_ptr<ngraph::Function> function) {
    DynamicToStaticShapeNonZero().run_on_function(function);

    return validateStaticShapes(function);
}

bool DynamicToStaticShape::validateStaticShapes(std::shared_ptr<ngraph::Function> function) const {
    function->validate_nodes_and_infer_types();

    for (const auto& node : function->get_ops()) {
        for (const auto& output : node->get_outputs()) {
            const auto outputPartialShape = output.get_partial_shape();
            VPU_THROW_UNLESS(outputPartialShape.is_static(),
                             "DynamicToStaticShape pass: after all the transformations there is "
                             "still dynamism in the network. First met node with dynamic output: "
                             "%s (type: %s)", node->get_friendly_name(), node->get_type_name());
            return false;
        }
    }
    return true;
}

}  // namespace pass
}  // namespace ngraph
