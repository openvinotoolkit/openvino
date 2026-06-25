// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace npuw {
namespace online {
class Snapshot;
}  // namespace online
namespace patterns {
namespace attn {

class GQA : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::GQA");
    static constexpr const char* pattern_name() {
        return "GQA";
    }
    static constexpr const char* isolation_tag() {
        return "attn";
    }
    static constexpr const char* group_name() {
        return "attn";
    }
    GQA(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

}  // namespace attn
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
