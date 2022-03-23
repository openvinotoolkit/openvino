// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface ConvertConstantsToScalars
 * @brief Replace only constants which are should be represented as scalars during code generation.
 * Only single-value (0D) constants are currently supported.
 * @ingroup snippets
 */
class ConvertConstants: public ngraph::pass::FunctionPass {
public:
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

} // namespace pass
} // namespace snippets
} // namespace ngraph