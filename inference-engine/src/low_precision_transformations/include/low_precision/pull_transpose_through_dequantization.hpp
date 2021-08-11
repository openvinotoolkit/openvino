// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API PullTransposeThroughDequantization;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

class ov::pass::low_precision::PullTransposeThroughDequantization : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PullTransposeThroughDequantization(const std::vector<ov::element::Type>& inputPrecisions = {});
};
