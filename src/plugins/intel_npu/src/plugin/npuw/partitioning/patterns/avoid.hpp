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

namespace online {
class Snapshot;  // Forward declaration
}  // namespace online

namespace patterns {
namespace avoid {

// Note: this pattern is only utilized by the online partitioner
class RMSNorm : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::avoid::RMSNorm");
    RMSNorm(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device);
};

}  // namespace avoid
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
