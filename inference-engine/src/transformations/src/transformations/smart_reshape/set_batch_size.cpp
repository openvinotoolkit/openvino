// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include <ngraph/pass/manager.hpp>

#include <transformations/init_node_info.hpp>
#include <transformations/itt.hpp>
#include <transformations/smart_reshape/mimic_set_batch_size.hpp>
#include <transformations/smart_reshape/set_batch_size.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::SetBatchSize, "SetBatchSize", 0);

bool ngraph::pass::SetBatchSize::run_on_function(std::shared_ptr<ngraph::Function> f) {
    OV_ITT_SCOPED_TASK(itt::domains::IETransform, "ngraph::pass::SetBatchSize");

    ngraph::pass::Manager manager;
    // This pass must be called first in pipeline
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::MimicSetBatchSize>();
    manager.run_passes(f);
    return true;
}

