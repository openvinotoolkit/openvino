// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace util {

class ShapeOfToConst : public ov::pass::MatcherPass {
public:
    ShapeOfToConst(const std::shared_ptr<ov::Model>& model);
};

}  // namespace util
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
