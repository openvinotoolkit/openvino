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
    IETRANSFORM_SCOPE(SmartReshape,
        ngraph::pass::Manager manager;

        // This pass must be called first in pipeline
        REGISTER_PASS(manager, InitNodeInfo);
        REGISTER_PASS(manager, ReshapeAMatMul);
        REGISTER_PASS(manager, ReshapeBMatMul);

        manager.run_passes(f);
        return true;
    )
    NGRAPH_CHECK(false, "nGraph pass is not included into the selective build.");
}
