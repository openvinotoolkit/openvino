// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "snippets/pass/base_tokenization_config.hpp"

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
class TokenizeMLPSeqSnippets : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::TokenizeMLPSeqSnippets");

    /**
     * @interface Config
     * @brief Configuration for TokenizeMLPSeqSnippets pass
     * @ingroup snippets
     */
    struct Config : public BaseTokenizationConfig {
        using CanBeFusedAsPostOpPred = std::function<bool(const std::shared_ptr<const ov::op::v0::MatMul>&,
                                                          const std::shared_ptr<const ov::Node>&)>;

        Config(size_t available_gprs_count,
               CanBeFusedAsPostOpPred can_be_fused_as_postop = nullptr)
            : BaseTokenizationConfig(available_gprs_count),
              m_can_be_fused_as_postop(std::move(can_be_fused_as_postop)) {
        }

        [[nodiscard]] const CanBeFusedAsPostOpPred& get_can_be_fused_as_postop() const {
            return m_can_be_fused_as_postop;
        }

    private:
        // Predicate that checks if the node can be fused as MatMul post-op.
        CanBeFusedAsPostOpPred m_can_be_fused_as_postop = nullptr;
    };

    explicit TokenizeMLPSeqSnippets(const Config& config);

private:
    static bool is_matmul_supported(const std::shared_ptr<ov::Node>& node);
    static bool is_supported_softmax(const std::shared_ptr<ov::Node>& node);
    static bool is_supported_intermediate_op(const std::shared_ptr<ov::Node>& node);
    static bool is_tensor_supported(const ov::descriptor::Tensor& t);

    static const size_t m_rank;
};

}  // namespace ov::snippets::pass
