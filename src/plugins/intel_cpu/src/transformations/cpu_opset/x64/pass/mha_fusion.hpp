// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include <openvino/opsets/opset4.hpp>

namespace ov {
namespace intel_cpu {

class MHAFusionBase : public ov::pass::MatcherPass {
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

class MHAFloatFusion: public MHAFusionBase {
public:
    OPENVINO_RTTI("MHAFloatFusion", "0");
    MHAFloatFusion();
};

class MHAFloatFusion2: public MHAFusionBase {
public:
    OPENVINO_RTTI("MHAFloatFusion2", "0");
    MHAFloatFusion2();
};

class MHAQuantFusion: public MHAFusionBase {
public:
    OPENVINO_RTTI("MHAQuantFusion", "0");
    MHAQuantFusion();
};

class MHAQuantFusion2: public MHAFusionBase {
public:
    OPENVINO_RTTI("MHAQuantFusion2", "0");
    MHAQuantFusion2();
};

class MHAFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("MHAFusion", "0");
    MHAFusion() {
        add_matcher<MHAFloatFusion>();
        add_matcher<MHAFloatFusion2>();
        add_matcher<MHAQuantFusion>();
        add_matcher<MHAQuantFusion2>();
    }
};

}   // namespace intel_cpu
}   // namespace ov
