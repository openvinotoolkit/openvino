// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_test_utils.hpp>

#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

class TokenizeMHASnippetsTests : public TransformationTestsF {
public:
    virtual void run();

protected:
    ov::snippets::pass::TokenizeMHASnippets::Config mha_config = get_default_mha_config();
    ov::snippets::pass::CommonOptimizations::Config common_config = get_default_common_optimizations_config();
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
