// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

class DecomposeTranspose : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DecomposeTranspose", "0");
    DecomposeTranspose();
};

class DecomposeTranspose2 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DecomposeTranspose2", "0");
    DecomposeTranspose2();
};

class DecomposeTranspose2FQ : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DecomposeTranspose2FQ", "0");
    DecomposeTranspose2FQ();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
