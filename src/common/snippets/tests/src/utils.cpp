// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <limits>

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
    static const CommonOptimizations::Config conf(1, true);
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
    static const TokenizeMLPSeqSnippets::Config conf(TokenizationConfig(std::numeric_limits<size_t>::max()), nullptr);
    return conf;
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
