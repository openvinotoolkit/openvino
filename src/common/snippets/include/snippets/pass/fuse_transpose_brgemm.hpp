// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface FuseTransposeBrgemm
 * @brief Fuses Transpose with Brgemm node, fusing on both Brgemm inputs and output is supported. Applicable to
 *        Transposes that don't change the position of the last dimension (since Brgemm supports strided rows i/o),
 *        but only 0213 Transpose is currently supported.
 * @ingroup snippets
 */
class FuseTransposeBrgemm: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseTransposeBrgemm", "0");
    FuseTransposeBrgemm();
    static const std::set<std::vector<int>> supported_cases;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph