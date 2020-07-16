// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/pass/graph_rewrite.hpp"

namespace vpu {

class MergeSubsequentDSROperations : public ngraph::pass::GraphRewrite {
public:
    MergeSubsequentDSROperations();
};

}  // namespace vpu
