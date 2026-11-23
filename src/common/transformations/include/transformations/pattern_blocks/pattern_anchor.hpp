// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string_view>

#include "openvino/core/node.hpp"

namespace ov::pass::pattern {

/// A named pattern node for generic inspection. The name must outlive the anchor.
struct PatternAnchor {
    std::string_view name;
    std::shared_ptr<Node> node;
};

}  // namespace ov::pass::pattern
