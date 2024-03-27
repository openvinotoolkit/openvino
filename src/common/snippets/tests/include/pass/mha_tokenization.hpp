// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_test_utils.hpp>

#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace test {
namespace snippets {

class TokenizeMHASnippetsTests : public TransformationTestsF {
public:
    virtual void run();

protected:
    ov::snippets::pass::SnippetsTokenization::Config config { 1, std::numeric_limits<size_t>::max(), true, true, { 3, 4 } };
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
