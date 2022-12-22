// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "shared_test_classes/base/snippets_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

class PrecisionPropagationTestValues{
public:
    ngraph::element::Type precision1;
    ngraph::PartialShape inputShape1;
    ngraph::element::Type precision2;
    ngraph::PartialShape inputShape2;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string, 
    PrecisionPropagationTestValues> PrecisionPropagationTestParams;

class PrecisionPropagationTest : 
    public testing::WithParamInterface<PrecisionPropagationTestParams>, 
    public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PrecisionPropagationTestParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
