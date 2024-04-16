// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_test_utils.hpp>

#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace test {
namespace snippets {

class TokenizeGNSnippetsTests : public TransformationTestsF {
public:
    virtual void run();
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
