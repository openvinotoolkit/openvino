// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

class U4BlockRepack : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pytorch::pass::U4BlockRepack");
    U4BlockRepack(bool is_symmetrical = false);
};

class U4ConvertReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pytorch::pass::U4ConvertReshape");
    U4ConvertReshape();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
