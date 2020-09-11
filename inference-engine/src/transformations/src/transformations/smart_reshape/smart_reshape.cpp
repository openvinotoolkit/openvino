// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "transformations/smart_reshape/smart_reshape.hpp"
#include "transformations/smart_reshape/reshape_with_hc_output.hpp"
#include "transformations/itt.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/init_node_info.hpp>

bool ngraph::pass::SmartReshape::run_on_function(std::shared_ptr<ngraph::Function> f) {
    OV_ITT_SCOPED_TASK(itt::domains::IETransform, "ngraph::pass::SmartReshape");

    ngraph::pass::Manager manager;
    // This pass must be called first in pipeline
    manager.register_pass<ngraph::pass::InitNodeInfo>();

    manager.register_pass<ngraph::pass::ReshapeAMatMul>();
    manager.register_pass<ngraph::pass::ReshapeBMatMul>();

    manager.run_passes(f);
    return true;
}
