// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace vpu {

class ConvertGatherND8ToGatherND5 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGatherND8ToGatherND5", "0");
    ConvertGatherND8ToGatherND5();
};
}
