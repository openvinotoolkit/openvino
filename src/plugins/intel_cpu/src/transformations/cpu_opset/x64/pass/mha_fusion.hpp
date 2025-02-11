// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/opsets/opset4.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu {

class MHAFusionBase : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MHAFusionBase");

protected:
    bool valid_transpose_order(const std::shared_ptr<ov::Node>& node, const std::vector<int64_t>& expected_order) {
        if (auto transpose_pattern = ov::as_type_ptr<ov::opset4::Constant>(node)) {
            if (transpose_pattern->cast_vector<int64_t>() != expected_order) {
                return false;
            }
        } else {
            return false;
        }

        return true;
    }
};

class MHAFloatFusion : public MHAFusionBase {
public:
    OPENVINO_RTTI("MHAFloatFusion", "0", MHAFusionBase);
    MHAFloatFusion();
};

class MHAFloatFusion2 : public MHAFusionBase {
public:
    OPENVINO_RTTI("MHAFloatFusion2", "0", MHAFusionBase);
    MHAFloatFusion2();
};

class MHAQuantFusion : public MHAFusionBase {
public:
    OPENVINO_RTTI("MHAQuantFusion", "0", MHAFusionBase);
    MHAQuantFusion();
};

class MHAQuantFusion2 : public MHAFusionBase {
public:
    OPENVINO_RTTI("MHAQuantFusion2", "0", MHAFusionBase);
    MHAQuantFusion2();
};

class MHAFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("MHAFusion");
    MHAFusion() {
        add_matcher<MHAFloatFusion>();
        add_matcher<MHAFloatFusion2>();
        add_matcher<MHAQuantFusion>();
        add_matcher<MHAQuantFusion2>();
    }
};

}  // namespace ov::intel_cpu
