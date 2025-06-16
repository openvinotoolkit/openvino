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

typedef std::tuple<ov::element::Type,  // Network Precision
                   std::string,        // Target Device
                   ov::AnyMap,         // Configuration
                   ov::Shape,          // Input Shapes
                   size_t              // Output Size
                   >
    matmulSqueezeAddParams;

class MatmulSqueezeAddTest : public testing::WithParamInterface<matmulSqueezeAddParams>,
                             virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<matmulSqueezeAddParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
