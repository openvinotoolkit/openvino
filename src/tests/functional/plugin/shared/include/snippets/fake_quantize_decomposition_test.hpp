// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {
class ActualValues {
public:
    ov::element::Type modelType;
    ov::Shape inputShape;
    ov::element::Type inputType;
    float zeroPoint;
    std::vector<ov::Shape> fakeQuantizeShapes;
};

class TestValues {
public:
    ov::element::Type modelType;
    ov::Shape inputShape;
    ov::element::Type inputType;
    float zeroPoint;
    std::vector<ov::Shape> fakeQuantizeShapes;
};

typedef std::tuple<
    TestValues,                 // test values
    std::pair<std::shared_ptr<ov::Node>, std::pair<std::string, std::string>>,   // operation
    std::pair<size_t, size_t>,  // number of nodes
    std::string                 // target device
> testsParams;

class FakeQuantizeDecompositionTest : public testing::WithParamInterface<testsParams>, virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<testsParams> obj);

protected:
    void SetUp() override;
};
}  // namespace snippets
}  // namespace test
}  // namespace ov
