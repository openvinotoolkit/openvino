// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "transformations/smart_reshape/smart_reshape.hpp"
#include "transformations/smart_reshape/matmul_sr.hpp"
#include "transformations/smart_reshape/softmax_sr.hpp"
#include "transformations/itt.hpp"

#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

bool ngraph::pass::SmartReshape::run_on_function(std::shared_ptr<ngraph::Function> f) {
    OV_ITT_SCOPED_TASK(itt::domains::IETransform, "ngraph::pass::SmartReshape");


    ngraph::pass::Manager smart_reshape_manager;
    // This pass must be called first in pipeline
    smart_reshape_manager.register_pass<ngraph::pass::InitNodeInfo>();

    auto doesnt_introduce_fake_dynamism = smart_reshape_manager.register_pass<ngraph::pass::GraphRewrite>();
    // fake dynamism here is shape dependence on the data
    // ShapeOf sub-graphs or even simple constant sub-graph may introduce such dynamism but it could be resolved by ConstantFolding
    doesnt_introduce_fake_dynamism->add_matcher<ngraph::pass::TransposeMatMul>();
    doesnt_introduce_fake_dynamism->set_name("ngraph::pass::SmartReshape::StaticShapeTransformations");

    auto introduces_fake_dynamism = smart_reshape_manager.register_pass<ngraph::pass::GraphRewrite>();
    introduces_fake_dynamism->add_matcher<ngraph::pass::ReshapeAMatMul>();
    introduces_fake_dynamism->add_matcher<ngraph::pass::ReshapeBMatMul>();
    introduces_fake_dynamism->add_matcher<ngraph::pass::ReshapeSoftMaxReshape>();
    introduces_fake_dynamism->set_name("ngraph::pass::SmartReshape::DynamicShapeTransformations");

    smart_reshape_manager.run_passes(f);
    return true;
}
