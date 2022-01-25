// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace MKLDNNPlugin {

class MarkupOptimalBS: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MarkupOptimalBS();
};

}  // namespace MKLDNNPlugin