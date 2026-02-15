// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_test_utils.hpp>

#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/mlp_seq_tokenization.hpp"
#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

class TokenizeMLPSeqSnippetsTests : public TransformationTestsF {
public:
    virtual void run();

protected:
    ov::snippets::pass::TokenizeMLPSeqSnippets::Config mlp_seq_config = get_default_mlp_seq_config();
    ov::snippets::pass::CommonOptimizations::Config common_config = get_default_common_optimizations_config();
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
