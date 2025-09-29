// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<std::vector<InputShape>,         // Input shapes (indices, values, dense_shape)
                   std::vector<float>,              // Default value
                   ov::element::Type,               // Data type for values and default value
                   ov::test::utils::InputLayerType, // Input layer type
                   ov::test::TargetDevice>          // Target device
    SparseFillEmptyRowsParams;

class SparseFillEmptyRowsLayerTest : public testing::WithParamInterface<SparseFillEmptyRowsParams>,
                                    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SparseFillEmptyRowsParams>& obj);
    using TGenData =
        testing::internal::CartesianProductHolder<testing::internal::ParamGenerator<std::vector<ov::test::InputShape>>,
                                                  testing::internal::ParamGenerator<std::vector<float>>,
                                                  testing::internal::ParamGenerator<ov::element::Type>,
                                                  testing::internal::ParamGenerator<ov::test::utils::InputLayerType>,
                                                  testing::internal::ValueArray<const char*>>;

    static const TGenData GetTestDataForDevice(const char* deviceName);
    static const TGenData GetStaticTestDataForDevice(const char* deviceName);
    static const TGenData GetDynamicTestDataForDevice(const char* deviceName);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};
}  // namespace test
}  // namespace ov
