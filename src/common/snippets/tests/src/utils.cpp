// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <limits>

#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/transpose.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/pass/mlp_seq_tokenization.hpp"
#include "snippets/pass/tokenization_config.hpp"

namespace ov {
namespace test {
namespace snippets {
using namespace ov::snippets::pass;

TokenizationConfig get_default_tokenization_config() {
    static const TokenizationConfig conf(std::numeric_limits<size_t>::max());
    return conf;
}

CommonOptimizations::Config get_default_common_optimizations_config() {
    static CommonOptimizations::Config conf(1, true);
    static bool initialized = false;
    if (!initialized) {
        conf.set_transpose_support_callback([](const std::shared_ptr<const ov::Node>& node) -> bool {
            const auto transpose = ov::as_type_ptr<const ov::op::v1::Transpose>(node->shared_from_this());
            if (!transpose) {
                return false;
            }
            const auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
            if (!order) {
                return false;
            }
            const auto order_value = order->cast_vector<int>();
            if (order_value.size() <= 2) {
                return false;
            }

            const auto& outputs = transpose->get_output_target_inputs(0);
            bool is_brgemm_case = false;
            if (!outputs.empty()) {
                const auto child_node = outputs.begin()->get_node()->shared_from_this();
                is_brgemm_case = ov::is_type<ov::op::v0::MatMul>(child_node);
            }
            return (is_brgemm_case && ov::snippets::pass::TokenizeMHASnippets::get_fusion_transpose_order(
                                          order_value.size()) == order_value) ||
                   (ov::snippets::pass::TokenizeMHASnippets::get_decomposed_transpose_order(order_value.size()) ==
                    order_value);
        });
        initialized = true;
    }
    return conf;
}

TokenizeMHASnippets::Config get_default_mha_config() {
    static const TokenizeMHASnippets::Config conf(TokenizationConfig(std::numeric_limits<size_t>::max()),
                                                  true,
                                                  true,
                                                  {3, 4});
    return conf;
}

TokenizeMLPSeqSnippets::Config get_default_mlp_seq_config() {
    static const TokenizeMLPSeqSnippets::Config conf(get_default_tokenization_config());
    return conf;
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
