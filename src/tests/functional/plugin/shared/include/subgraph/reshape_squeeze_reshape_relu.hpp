// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using ShapeAxesTuple = std::pair<ov::Shape, std::vector<int>>;

using ReshapeSqueezeReshapeReluTuple = typename std::tuple<ShapeAxesTuple,     // Input shapes & squeeze_indices
                                                           ov::element::Type,  // Input element type
                                                           std::string,        // Device name
                                                           ov::test::utils::SqueezeOpType  // SqueezeOpType
                                                           >;

class ReshapeSqueezeReshapeRelu : public testing::WithParamInterface<ReshapeSqueezeReshapeReluTuple>,
                                  virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReshapeSqueezeReshapeReluTuple>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
