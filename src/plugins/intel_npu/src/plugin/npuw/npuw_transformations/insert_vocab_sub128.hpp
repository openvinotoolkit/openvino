// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov::npuw {

namespace vocab_sub128 {

inline constexpr const char* marker = "npuw::vocab_sub128";
inline constexpr const char* friendly_name_suffix = "_npuw_vocab_sub128";

inline void mark(const std::shared_ptr<ov::Node>& node) {
    node->get_rt_info()[marker] = true;
    node->set_friendly_name(node->get_friendly_name() + friendly_name_suffix);
}

}  // namespace vocab_sub128

class InsertVocabSub128 : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::npuw::InsertVocabSub128");
    InsertVocabSub128();
};

}  // namespace ov::npuw
