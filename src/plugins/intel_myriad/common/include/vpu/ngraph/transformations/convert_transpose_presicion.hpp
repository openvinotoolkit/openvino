// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace vpu {

class ConvertTransposePrecision : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertTransposePrecision", "0");
    ConvertTransposePrecision();
};
}
