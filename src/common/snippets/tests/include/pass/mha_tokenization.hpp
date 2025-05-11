// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_test_utils.hpp>

#include "snippets/pass/tokenization.hpp"
#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

class TokenizeMHASnippetsTests : public TransformationTestsF {
public:
    virtual void run();

protected:
    ov::snippets::pass::SnippetsTokenization::Config config = get_default_tokenization_config();
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
