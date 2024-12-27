// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class QKVProjFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("QKVProjFusion");
    QKVProjFusion();
};

class QKVProjFusion2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("QKVProjFusion2");
    QKVProjFusion2();
};

}  // namespace intel_cpu
}  // namespace ov
