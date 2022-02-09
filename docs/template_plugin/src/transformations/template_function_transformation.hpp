// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/pass.hpp>

namespace ngraph {
namespace pass {

class MyFunctionTransformation;

}  // namespace pass
}  // namespace ngraph

// ! [function_pass:template_transformation_hpp]
// template_function_transformation.hpp
class ngraph::pass::MyFunctionTransformation : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;
};
// ! [function_pass:template_transformation_hpp]
