// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/pass.hpp>

// ! [function_pass:template_transformation_hpp]
// template_function_transformation.hpp
class MyFunctionTransformation: public ngraph::pass::FunctionPass {
public:
    MyFunctionTransformation() : FunctionPass() {}

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
// ! [function_pass:template_transformation_hpp]