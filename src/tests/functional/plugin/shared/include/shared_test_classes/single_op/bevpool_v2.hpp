// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using BevPoolV2Params = std::tuple<std::vector<InputShape>,
                                   ov::element::Type,  // feature type (cf, dw)
                                   ov::element::Type,  // index type (idx, itv)
                                   ov::test::TargetDevice>;

class BevPoolV2LayerTest : public testing::WithParamInterface<BevPoolV2Params>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BevPoolV2Params>& obj);

    using TGenData = testing::internal::CartesianProductHolder<
        testing::internal::ParamGenerator<std::vector<ov::test::InputShape>>,
        testing::internal::ParamGenerator<ov::element::Type>,
        testing::internal::ParamGenerator<ov::element::Type>,
        testing::internal::ValueArray<const char*>>;

    static const TGenData GetTestDataForDevice(const char* deviceName);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

}  // namespace test
}  // namespace ov
