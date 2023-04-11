// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

class ConstantConvertFoldingTests : public TransformationTestsF {
public:
    virtual void run();
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
