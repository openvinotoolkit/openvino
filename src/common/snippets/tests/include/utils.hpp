// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/pass/base_tokenization_config.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/pass/mlp_seq_tokenization.hpp"

namespace ov {
namespace test {
namespace snippets {

ov::snippets::pass::TokenizationConfig get_default_tokenization_config();
ov::snippets::pass::CommonOptimizations::Config get_default_common_optimizations_config();
ov::snippets::pass::TokenizeMHASnippets::Config get_default_mha_config();
ov::snippets::pass::TokenizeMLPSeqSnippets::Config get_default_mlp_seq_config();

}  // namespace snippets
}  // namespace test
}  // namespace ov
