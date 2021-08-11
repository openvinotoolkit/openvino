// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>


namespace ov {
namespace pass {

class TRANSFORMATIONS_API SmartReshape;

}  // namespace pass
}  // namespace ov

class ov::pass::SmartReshape: public ov::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ov::Function> f) override;
};
