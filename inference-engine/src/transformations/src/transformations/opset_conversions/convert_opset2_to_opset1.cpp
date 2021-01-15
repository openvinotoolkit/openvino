// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/opset_conversions/convert_opset2_to_opset1.hpp"

#include "transformations/op_conversions/convert_batch_to_space.hpp"
#include "transformations/op_conversions/convert_space_to_batch.hpp"
#include "itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertOpSet2ToOpSet1, "ConvertOpSet2ToOpSet1", 0);

bool ngraph::pass::ConvertOpSet2ToOpSet1::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(ConvertOpSet2ToOpSet1);
    ngraph::pass::Manager manager(get_pass_config());

    manager.register_pass<ngraph::pass::ConvertSpaceToBatch>();
    manager.register_pass<ngraph::pass::ConvertBatchToSpace>();

    manager.run_passes(f);
    return true;
}
