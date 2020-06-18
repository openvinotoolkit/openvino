// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "template_function_transformation.hpp"

#include <ngraph/opsets/opset3.hpp>

// ! [function_pass:template_transformation_cpp]
// template_function_transformation.cpp
bool MyFunctionTransformation::run_on_function(std::shared_ptr<ngraph::Function> f) {
    // Transformation code
    return false;
}
// ! [function_pass:template_transformation_cpp]