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
typedef std::tuple<std::vector<InputShape>,
                   int64_t,            // frame size value
                   int64_t,            // frame step value
                   int64_t,            // signal length value (optional, set test param to -1 to use default)
                   bool,               // center
                   bool,               // normalized
                   ov::element::Type,  // data type
                   ov::element::Type,  // size/step type
                   ov::test::utils::InputLayerType,
                   ov::test::TargetDevice>
    ISTFTParams;

class ISTFTLayerTest : public testing::WithParamInterface<ISTFTParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ISTFTParams>& obj);
    using TGenData =
        testing::internal::CartesianProductHolder<testing::internal::ParamGenerator<std::vector<ov::test::InputShape>>,
                                                  testing::internal::ParamGenerator<int64_t>,
                                                  testing::internal::ParamGenerator<int64_t>,
                                                  testing::internal::ParamGenerator<int64_t>,
                                                  testing::internal::ParamGenerator<bool>,
                                                  testing::internal::ParamGenerator<bool>,
                                                  testing::internal::ParamGenerator<ov::element::Type>,
                                                  testing::internal::ParamGenerator<ov::element::Type>,
                                                  testing::internal::ParamGenerator<ov::test::utils::InputLayerType>,
                                                  testing::internal::ValueArray<const char*>>;

    static const TGenData GetTestDataForDevice(const char* deviceName);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
