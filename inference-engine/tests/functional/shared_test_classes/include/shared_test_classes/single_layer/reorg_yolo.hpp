// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

using ReorgYoloParamsTuple = typename std::tuple<InputShape,     // Input Shape
                                                 size_t,         // stride
                                                 ElementType,    // Network precision
                                                 TargetDevice>;  // Device

class ReorgYoloLayerTest : public testing::WithParamInterface<ReorgYoloParamsTuple>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReorgYoloParamsTuple>& obj);

protected:
    void SetUp() override;
};

}  // namespace subgraph
}  // namespace test
}  // namespace ov
