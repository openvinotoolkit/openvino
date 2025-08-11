// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace test {
struct RandomUniformTypeSpecificParams {
    ov::element::Type model_type;       // Model type
    double min_value;                   // Min value constant, will be cast to the needed precision
    double max_value;                   // Max value constant, will be cast to the needed precision
};

using RandomUniformParamsTuple = typename std::tuple<
    ov::Shape,                          // Input shape
    RandomUniformTypeSpecificParams,    // Parameters which depends on output type
    int64_t,                            // Global seed
    int64_t,                            // Operation seed
    ov::op::PhiloxAlignment,            // Alignment of generator
    ov::test::TargetDevice              // Device name
>;

class RandomUniformLayerTest : public testing::WithParamInterface<RandomUniformParamsTuple>,
                               virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RandomUniformParamsTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov

