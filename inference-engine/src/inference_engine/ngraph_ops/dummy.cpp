// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "dummy.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

op::Dummy::Dummy(): Op("Dummy", {}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::Dummy::copy_with_new_args(const NodeVector& new_args) const {
    if (!new_args.empty())
        throw ngraph_error("Incorrect number of new arguments");

    return make_shared<Dummy>();
}

void op::Dummy::validate_and_infer_types() {
    set_output_type(0, ngraph::element::Type(), {});
}

