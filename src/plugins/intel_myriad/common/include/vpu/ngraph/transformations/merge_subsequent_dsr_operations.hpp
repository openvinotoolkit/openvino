// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/pass/graph_rewrite.hpp"

namespace vpu {

class MergeSubsequentDSROperations : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MergeSubsequentDSROperations", "0");
    MergeSubsequentDSROperations();
};

}  // namespace vpu
