// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<InputShape>,            // Input Shapes
        std::pair<bool, bool>,              // Transpose inputs
        ov::element::Type,                  // Model type
        ov::test::utils::InputLayerType,    // Secondary input type
        std::string,                        // Device name
        std::map<std::string, std::string>  // Additional network configuration
> MatMulLayerTestParamsSet;

class MatMulLayerTest : public testing::WithParamInterface<MatMulLayerTestParamsSet>,
                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulLayerTestParamsSet> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
