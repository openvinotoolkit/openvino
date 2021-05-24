// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/validate.hpp"
#include "itt.hpp"
#include "ngraph/graph_util.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::Validate, "ngraph::pass::Validate", 0);

bool pass::Validate::run_on_function(std::shared_ptr<Function> f)
{
    f->validate_nodes_and_infer_types();
    return false;
}
