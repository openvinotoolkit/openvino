// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <map>

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
using InputShapesTuple = std::pair<std::vector<size_t>, std::vector<size_t>>;

typedef std::tuple<
    std::vector<InputShape>,            // Input shapes
    ov::test::utils::LogicalTypes,      // Logical op type
    ov::test::utils::InputLayerType,    // Second input type
    ov::element::Type,                  // Model type
    std::string,                        // Device name
    std::map<std::string, std::string>  // Additional model configuration
> LogicalTestParams;

class LogicalLayerTest : public testing::WithParamInterface<LogicalTestParams>,
    virtual public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<LogicalTestParams>& obj);
};
} // namespace test
} // namespace ov
