// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
class AssignRegisters : public ngraph::pass::FunctionPass {
public:
    AssignRegisters() {
        set_property(ngraph::pass::PassProperty::REQUIRE_STATIC_SHAPE, true);
    }
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
