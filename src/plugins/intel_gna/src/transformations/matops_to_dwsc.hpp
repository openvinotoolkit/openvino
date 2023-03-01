// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

class AddDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("AddDecomposition", "0");
    AddDecomposition();
};

class SubDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SubDecomposition", "0");
    SubDecomposition();
};

class MulDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MulDecomposition", "0");
    MulDecomposition();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
