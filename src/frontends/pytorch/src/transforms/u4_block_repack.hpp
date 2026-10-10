// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::frontend::pytorch::pass {

class U4BlockRepack : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::U4BlockRepack");
    U4BlockRepack(bool is_symmetrical = false);
};

class U4ConvertReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::U4ConvertReshape");
    U4ConvertReshape();
};

}  // namespace ov::frontend::pytorch::pass
