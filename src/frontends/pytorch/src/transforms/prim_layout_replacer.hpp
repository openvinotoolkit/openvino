// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

class PrimLayoutReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PrimLayoutReplacer", "0");
    PrimLayoutReplacer();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
