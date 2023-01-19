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
 * @interface MatMulToBrgemm
 * @brief Replaces ngraph::MatMul with snippets::op::Brgemm operation (only non-trasposing MatMuls are currently supported)
 * @ingroup snippets
 */
class MatMulToBrgemm: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MatMulToBrgemm", "0");
    MatMulToBrgemm();
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
