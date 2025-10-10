// Copyright (C) 2025 Intel Corporation
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

// Note: the patterns below are only utilized by the online partitioner
namespace attn {

class SDPA : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::SDPA");
    SDPA(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

}  // namespace attn

namespace regularize {

class AttentionBroadcast : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::AttentionBroadcast");
    AttentionBroadcast();
};

class ShapeOfParameter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::ShapeOfParameter");
    ShapeOfParameter();
};

}  // namespace regularize

}  // namespace patterns
}  // namespace npuw
}  // namespace ov
