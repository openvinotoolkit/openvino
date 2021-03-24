// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/pass.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface AssignRegisters
 * @brief Assigns internal `vector` register indexes to operations.
 * Changing order of variables or datafrow lead to invalidation of register assignment.
 * @ingroup snippets
 */
class TRANSFORMATIONS_API AssignRegisters : public ngraph::pass::FunctionPass {
public:
    AssignRegisters() : FunctionPass() {
        set_property(ngraph::pass::PassProperty::REQUIRE_STATIC_SHAPE, true);
    }
    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
