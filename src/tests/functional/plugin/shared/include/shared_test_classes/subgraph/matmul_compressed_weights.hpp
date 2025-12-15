// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<ov::element::Type,  // Input Precision
                   ov::element::Type,  // Weights Precision
                   std::string,        // Target Device
                   ov::AnyMap,         // Configuration
                   ov::Shape,          // Input Shape
                   ov::Shape>          // Weights Shape
    matmulCompressedParams;

class MatmulCompressedTest : public testing::WithParamInterface<matmulCompressedParams>,
                             virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<matmulCompressedParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
