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


/**
 * @interface InsertReverseConvert
 * @brief After FakeQuantize there are ConvertSaturations from F32 to U8 inside body. This pass inserts reverse ConvertSaturations
 *        after them to return FP32 calculation inside bofy if these original ConvertSaturations aren't on Results
 * @ingroup snippets
 */
class InsertReverseConvert: public ngraph::pass::MatcherPass {
public:
    InsertReverseConvert();
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
