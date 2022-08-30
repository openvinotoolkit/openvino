// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface InsertConvertOnInputs
 * @brief Inserts ConvertSaturation op after Parameters and Scalars to convert data type of inputs
 *        to supported execution data type.
 *        Note: ConvertSaturation op isn't covered by specification of "Convert" op
 *              This op is used for conversion into and from FP32 after the correspoding Load
 *              and before Store to calculate in FP32 inside subgraph body in CPU Plugin
 * @ingroup snippets
 */
class InsertConvertOnInputs: public ngraph::pass::MatcherPass {
public:
    InsertConvertOnInputs(const ov::element::Type exec_type = ov::element::f32);
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
