// Copyright (C) 2018-2026 Intel Corporation
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
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::U4BlockRepack");
    U4BlockRepack(bool is_symmetrical = false);
};

class U4ConvertReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::U4ConvertReshape");
    U4ConvertReshape();
};

class U2ConvertReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::U2ConvertReshape");
    U2ConvertReshape();
};

/// \brief Marks Convert nodes consuming u2/u4 Constants with disable_constant_folding
///        and mark_as_decompression to prevent MOC from folding compressed weight constants.
class MarkCompressedWeightConstants : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::MarkCompressedWeightConstants");
    MarkCompressedWeightConstants();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
