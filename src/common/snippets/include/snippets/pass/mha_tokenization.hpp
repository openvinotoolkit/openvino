// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "snippets/pass/base_tokenization_config.hpp"

namespace ov::snippets::pass {

/**
 * @interface TokenizeMHASnippets
 * @brief The pass tokenizes MHA-pattern into Subgraph
 *        Pattern:           Transpose1
 *                               |
 *             Transpose0 [Eltwise, Select]
 *                     \     /
 *                     MatMul0
 *                        |
 *           [Eltwise, Select, Reshape]
 *                        |
 *                     Softmax
 *                        |
 *            [Eltwise, Select, Reshape]  Transpose2
 *                               \      /
 *                                MatMul1
 *                                  |
 *                  [Eltwise, Select, Transpose3]
 *        Notes:
 *          - Transposes can be missed
 *          - Transpose0, Transpose2 and Transpose3 may have only [0,2,1,3] order
 *          - Transpose1 may have only [0,2,3,1] order
 *          - [...] means any count of different nodes from list. But:
 *              * Reshapes can be only explicitly around Softmax (Reshape -> Softmax -> Reshape)
 *              * After MatMul1 may be only Transpose3 or any count of Eltwise, Select ops.
 * @ingroup snippets
 */
class TokenizeMHASnippets : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::TokenizeMHASnippets");

    /**
     * @interface Config
     * @brief Configuration for TokenizeMHASnippets pass
     * @ingroup snippets
     */
    struct Config : public BaseTokenizationConfig {
        Config(size_t available_gprs_count,
               bool enable_transpose_on_output,
               bool dyn_mha_token,
               std::set<size_t> mha_transpose_ranks)
            : BaseTokenizationConfig(available_gprs_count),
              m_mha_token_enable_transpose_on_output(enable_transpose_on_output),
              m_is_dynamic_mha_token_enabled(dyn_mha_token),
              m_mha_supported_transpose_ranks(std::move(mha_transpose_ranks)) {
        }

        [[nodiscard]] bool get_mha_token_enable_transpose_on_output() const {
            return m_mha_token_enable_transpose_on_output;
        }

        [[nodiscard]] bool is_dynamic_mha_token_enabled() const {
            return m_is_dynamic_mha_token_enabled;
        }

        [[nodiscard]] std::set<size_t> get_mha_supported_transpose_ranks() const {
            return m_mha_supported_transpose_ranks;
        }

    private:
        // False if Transpose on output isn't tokenized in MHA Tokenization.
        // Otherwise, it may be fused into Subgraph if possible
        // TODO [111813]: Remove please when the ticket 111813 is implemented
        bool m_mha_token_enable_transpose_on_output = true;
        // If True, MHA pattern with dynamic nodes will be tokenized
        // Otherwise dynamic MHA won't be tokenized
        // Currently, the flag can be set to `True` only for testing purposes.
        bool m_is_dynamic_mha_token_enabled = true;
        // Set of supported Transpose shape ranks for tokenization in MHATokenization pass.
        // Note that in general Snippets support Transpose of any ranks.
        // But at the moment Transpose is used only in MHA pattern where 3D and 4D tensors are supported.
        std::set<size_t> m_mha_supported_transpose_ranks = {3, 4};
    };

    explicit TokenizeMHASnippets(const Config& config);

    static std::vector<int32_t> get_fusion_transpose_order(size_t rank);
    static std::vector<int32_t> get_decomposed_transpose_order(size_t rank);
    static bool is_matmul0_supported(const std::shared_ptr<ov::opset1::MatMul>& matmul);
};

}  // namespace ov::snippets::pass
