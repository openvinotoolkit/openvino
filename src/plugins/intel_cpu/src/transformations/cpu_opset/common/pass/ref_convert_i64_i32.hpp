// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

// This pass inserts Convert node from i64 to i32 for Reference nodes.

class RefConvertI64ToI32: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RefConvertI64ToI32", "0");
    RefConvertI64ToI32();
};

}  // namespace pass
}  // namespace ov
