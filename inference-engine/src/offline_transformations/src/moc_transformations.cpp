// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "moc_transformations.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::MOCTransformations, "MOCTransformations", 0);

bool ngraph::pass::MOCTransformations::run_on_function(std::shared_ptr<ngraph::Function>) {
    return false;
}