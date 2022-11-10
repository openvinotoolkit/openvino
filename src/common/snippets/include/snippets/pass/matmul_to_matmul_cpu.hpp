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
class MatMulToMatMulCPU: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MatMulToMatMulCPU", "0");
    MatMulToMatMulCPU();
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
