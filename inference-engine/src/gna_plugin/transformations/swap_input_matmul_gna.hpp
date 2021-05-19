// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

// @brief Swaps and transposes inputs of MatMul if its first input is const and its batch size isn't supported by GNA
class SwapInputMatMul: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SwapInputMatMul();
};
} // namespace GNAPluginNS