// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include <ngraph/pass/manager.hpp>

#include <transformations/init_node_info.hpp>
#include <transformations/itt.hpp>
#include <transformations/smart_reshape/reshape_to_1D.hpp>
#include <transformations/smart_reshape/reshape_with_hc_output.hpp>
#include <transformations/smart_reshape/smart_reshape.hpp>
#include <transformations/smart_reshape/proposal_scales_stridedslice.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::SmartReshape, "SmartReshape", 0);

bool ngraph::pass::SmartReshape::run_on_function(std::shared_ptr<ngraph::Function> f) {
    OV_ITT_SCOPED_TASK(itt::domains::IETransform, "ngraph::pass::SmartReshape");

    ngraph::pass::Manager static_manager;
    // This pass must be called first in pipeline
    static_manager.register_pass<ngraph::pass::InitNodeInfo>();
    static_manager.register_pass<ngraph::pass::ReshapeTo1D>();
    static_manager.register_pass<ngraph::pass::opset1_ProposalScales>();
    static_manager.register_pass<ngraph::pass::opset4_ProposalScales>();
    static_manager.run_passes(f);

    ngraph::pass::Manager dynamic_manager;
    dynamic_manager.set_per_pass_validation(false);
    dynamic_manager.register_pass<ngraph::pass::ReshapeAMatMul>();
    dynamic_manager.register_pass<ngraph::pass::ReshapeBMatMul>();
    dynamic_manager.run_passes(f);
    return true;
}
