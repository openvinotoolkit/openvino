// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

using SoftMaxTestParams = std::tuple<
        ElementType,                                    // netPrecision
        ElementType,                                    // inPrecision
        ElementType,                                    // outPrecision
        InputShape,                                     // Dynamic shape + Target static shapes
        size_t,                                         // axis
        TargetDevice,                                   // targetDevice
        Config                                          // config
>;

class SoftMaxLayerTest : public testing::WithParamInterface<SoftMaxTestParams>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SoftMaxTestParams> &obj);

protected:
    void SetUp() override;
};
}  // namespace subgraph
}  // namespace test
}  // namespace ov
