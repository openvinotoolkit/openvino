// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        ov::element::Type_t,  // Network Precision
        InputShape,           // Input1 Shape,
        InputShape,           // Input2 Shape,
        bool,
        std::string           // Target Device
> CodegenGeluParams;

class CodegenGelu : public testing::WithParamInterface<ov::test::snippets::CodegenGeluParams>,
                    virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::CodegenGeluParams> obj);

protected:
    void SetUp() override;
};


} // namespace snippets
} // namespace test
} // namespace ov
