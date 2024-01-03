// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "ov_models/builders.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

using MultinomialTestParams = std::tuple<
    ElementType,        // netPrecision
    ElementType,        // inPrecision
    ElementType,        // outPrecision
    InputShape,         // Dynamic shape + Target static shapes
    std::int64_t,       // Number of samples
    element::Type_t,    // Output type attribute
    bool,               // With replacement,
    bool,               // Log probs;
    TargetDevice,       // targetDevice
    Config              // config
    >;

class MultinomialTest : public testing::WithParamInterface<MultinomialTestParams>,
                               virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MultinomialTestParams> &obj);

protected:
    void SetUp() override;
};

} // namespace subgraph
} // namespace test
} // namespace ov
