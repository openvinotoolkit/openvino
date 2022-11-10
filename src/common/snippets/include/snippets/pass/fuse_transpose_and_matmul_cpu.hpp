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
 * @interface FuseLoadConvert
 * @brief Fuse Load and ConvertSaturation into one op LoadConvertSaturation
 *        Fuse Load and ConvertTruncation into one op LoadConvertTruncation
 * @ingroup snippets
 */
class FuseTransposeMatMulCPU: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseTransposeMatMulCPU", "0");
    FuseTransposeMatMulCPU();
    static const std::set<std::vector<int>> supported_cases;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph