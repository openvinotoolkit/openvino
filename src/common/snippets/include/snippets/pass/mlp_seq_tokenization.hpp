// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "snippets/pass/tokenization_config.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::pass {

/**
 * @interface TokenizeMLPSeqSnippets
 * @brief The pass tokenizes sequence of MLPs into Subgraph
 *        Pattern:
 * [Eltwise, FQ, Softmax]  Constant
 *              |         /
 *           MatMul0
 *              |
 * [Eltwise, FQ, Softmax]  Constant
 *              |        /
 *           MatMul1
 *              |
 *             ...
 * @ingroup snippets
 */
class SNIPPETS_API TokenizeMLPSeqSnippets : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::TokenizeMLPSeqSnippets");

    /**
     * @interface Config
     * @brief Configuration for TokenizeMLPSeqSnippets pass
     * @ingroup snippets
     */
    struct SNIPPETS_API Config : public TokenizationConfig {
        using CanBeFusedAsPostOpPred = std::function<bool(const std::shared_ptr<const ov::op::v0::MatMul>&,
                                                          const std::shared_ptr<const ov::Node>&)>;

        static bool postops_are_not_supported([[maybe_unused]] const std::shared_ptr<const ov::op::v0::MatMul>& matmul,
                                              [[maybe_unused]] const std::shared_ptr<const ov::Node>& postop) {
            return false;
        }

        explicit Config(const TokenizationConfig& tokenization_config,
                        CanBeFusedAsPostOpPred can_be_fused_as_postop = postops_are_not_supported)
            : TokenizationConfig(tokenization_config),
              m_can_be_fused_as_postop(std::move(can_be_fused_as_postop)) {}

        [[nodiscard]] const CanBeFusedAsPostOpPred& get_can_be_fused_as_postop() const {
            OPENVINO_ASSERT(m_can_be_fused_as_postop, "m_can_be_fused_as_postop mustn't be nullptr");
            return m_can_be_fused_as_postop;
        }

    private:
        // Predicate that checks if the node can be fused as MatMul post-op.
        CanBeFusedAsPostOpPred m_can_be_fused_as_postop = postops_are_not_supported;
    };

    explicit TokenizeMLPSeqSnippets(const Config& config);

private:
    static bool is_matmul_supported(const std::shared_ptr<ov::Node>& node);
    static bool is_supported_softmax(const std::shared_ptr<ov::Node>& node);
    static bool is_supported_intermediate_op(const std::shared_ptr<ov::Node>& node);
    static bool is_tensor_supported(const ov::descriptor::Tensor& t);

    static constexpr size_t m_rank = 2;
};

}  // namespace ov::snippets::pass
