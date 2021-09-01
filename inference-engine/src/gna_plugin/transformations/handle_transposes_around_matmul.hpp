// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ie/ie_common.h>
#include "backend/gna_limitations.hpp"

namespace GNAPluginNS {

class HandleTransposeAfterMatMulWithLastReshape;
class HandleTransposeAfterMatMulWithLastTranspose;

namespace {
struct VerifyReshape {
    bool operator()(const ngraph::Output<ngraph::Node>& reshape_out) const;
}; // struct VerifyReshape

struct Helper {
    static void ReplaceTransposeWithReshape(std::shared_ptr<ngraph::Node> transpose_node);
    static void InsertTranspose(std::shared_ptr<ngraph::Node> prev_node, const std::string& base_name);
    template<typename T>
    static bool CreateMatcher(std::shared_ptr<ngraph::pattern::Matcher>& matcher, ngraph::graph_rewrite_callback& callback) {
        auto matmul = ngraph::pattern::wrap_type<ngraph::opset8::MatMul>();
        auto add_left = ngraph::pattern::wrap_type<ngraph::opset8::Add>({matmul, ngraph::pattern::any_input()});
        auto add_right = ngraph::pattern::wrap_type<ngraph::opset8::Add>({ngraph::pattern::any_input(), matmul});
        auto fq_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{matmul, add_left, add_right});
        auto fq = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({fq_input, ngraph::pattern::any_input(),
            ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
        auto transpose_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{fq_input, fq});
        auto transpose = ngraph::pattern::wrap_type<ngraph::opset8::Transpose>({transpose_input, ngraph::pattern::any_input()});
        auto reshape_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{transpose_input, transpose});
        auto reshape = ngraph::pattern::wrap_type<ngraph::opset8::Reshape>(
            {reshape_input, ngraph::pattern::any_input()}, VerifyReshape());
        callback = [=](ngraph::pattern::Matcher &m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto matmul_name = pattern_map.at(matmul).get_node_shared_ptr()->get_friendly_name();
            auto transpose_it = pattern_map.find(transpose);
            if (transpose_it != std::end(pattern_map)) {
                if (std::is_same<T, HandleTransposeAfterMatMulWithLastTranspose>::value) {
                    for (const auto& output : transpose_it->second.get_node_shared_ptr()->outputs()) {
                        for (const auto& input : output.get_target_inputs()) {
                            if (dynamic_cast<ngraph::opset8::Reshape*>(input.get_node())) {
                                return false;
                            }
                        }
                    }
                }

                ReplaceTransposeWithReshape(transpose_it->second.get_node_shared_ptr());
            } else {
                auto iter = pattern_map.find(reshape);
                if (iter != pattern_map.end() &&
                    !GNALimitations::IsTransposeSupported(iter->second.get_node_shared_ptr()->get_input_shape(0))) {
                    return false;
                }
                std::shared_ptr<ngraph::Node> node;
                if ((iter = pattern_map.find(fq)) != pattern_map.end()) {
                    node = iter->second.get_node_shared_ptr();
                } else if ((iter = pattern_map.find(add_left)) != pattern_map.end()) {
                    node = iter->second.get_node_shared_ptr();
                } else if ((iter = pattern_map.find(add_right)) != pattern_map.end()) {
                    node = iter->second.get_node_shared_ptr();
                } else {
                    node = pattern_map.at(matmul).get_node_shared_ptr();
                }

                InsertTranspose(node, node->get_friendly_name());
            }
            return true;
        };

        if (std::is_same<T, HandleTransposeAfterMatMulWithLastTranspose>::value) {
            matcher = std::make_shared<ngraph::pattern::Matcher>(transpose, T::Name());
            return true;
        }

        if (std::is_same<T, HandleTransposeAfterMatMulWithLastReshape>::value) {
            matcher = std::make_shared<ngraph::pattern::Matcher>(reshape, T::Name());
            return true;
        }

        return false;
    }
}; // struct Helper
} // namespace

/**
 * @brief Inserts Transpose before MatMul or removes it (if it exists) if there is Reshape
 * before MatMul which changes the batch size:
 *    [1, A*B]                 [1, A*B]
 *       |                       |
 *    Reshape                 Reshape
 *       |                       |
 * [1, A, 1, B]            [1, A, 1, B]
 *       |                       |
 *       |                   Transpose
 *       |           ->          |
 *       |           <-     [1, B, 1, A]
 *       |                       |
 *    MatMul                   MatMul
 */
class HandleTransposeBeforeMatMul : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  HandleTransposeBeforeMatMul();
};

/**
 * @brief Inserts Transpose after MatMul or removes it (if it exists) if there is Reshape
 * after MatMul which changes the batch size:
 *    MatMul                  MatMul
 *       |                       |
 * [1, A, 1, B]            [1, A, 1, B]
 *       |                       |
 *       |                   Transpose
 *       |           ->          |
 *       |           <-     [1, B, 1, A]
 *       |                       |
 *    Reshape                 Reshape
 *       |                       |
 *    [1, A*B]                [1, A*B]
 */
class HandleTransposeAfterMatMulWithLastReshape: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HandleTransposeAfterMatMulWithLastReshape();
    static std::string Name() { return "HandleTransposeAfterMatMulWithLastReshape"; }
};

class HandleTransposeAfterMatMulWithLastTranspose: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HandleTransposeAfterMatMulWithLastTranspose();
    static std::string Name() { return "HandleTransposeAfterMatMulWithLastTranspose"; }
};

class HandleTransposesAroundMatMul : public ngraph::pass::GraphRewrite {
public:
  NGRAPH_RTTI_DECLARATION;
  HandleTransposesAroundMatMul();
};

} // namespace GNAPluginNS
