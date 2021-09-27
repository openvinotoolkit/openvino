// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/validate.hpp"

#include "itt.hpp"
#include "ngraph/graph_util.hpp"

using namespace ngraph;

bool ov::pass::Validate::run_on_function(std::shared_ptr<Function> f) {
    f->validate_nodes_and_infer_types();
    return false;
}
