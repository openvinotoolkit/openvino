// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {

// Model optimization patterns. Triggered by the plugin at the very top
namespace patterns {
namespace opt {

class DQMatMulCWi : public ov::pass::MatcherPass {
public:
    DQMatMulCWi();
};

}  // namespace opt
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
