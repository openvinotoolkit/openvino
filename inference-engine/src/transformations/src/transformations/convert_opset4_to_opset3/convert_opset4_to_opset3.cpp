// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset4_to_opset3/convert_opset4_to_opset3.hpp"

#include "transformations/convert_gelu.hpp"
#include "transformations/convert_batch_to_space.hpp"
#include "transformations/convert_space_to_batch.hpp"
#include "transformations/itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertOpSet4ToOpSet3, "ConvertOpSet4ToOpSet3", 0);

bool ngraph::pass::ConvertOpSet4ToOpSet3::run_on_function(std::shared_ptr<ngraph::Function> f) {
    OV_ITT_SCOPED_TASK(itt::domains::IETransform, "ngraph::pass::ConvertOpSet4ToOpSet3");

    ngraph::pass::Manager manager;

    manager.register_pass<ngraph::pass::ConvertSpaceToBatch>();
    manager.register_pass<ngraph::pass::ConvertBatchToSpace>();

    manager.set_callback(m_transformation_callback);
    manager.run_passes(f);
    return true;
}
