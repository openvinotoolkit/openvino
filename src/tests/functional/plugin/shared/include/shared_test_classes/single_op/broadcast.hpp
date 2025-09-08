// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using BroadcastParamsTuple = typename std::tuple<
        std::vector<size_t>,       // target shape
        ov::AxisSet,               // axes mapping
        ov::op::BroadcastType,     // broadcast mode
        std::vector<InputShape>,   // Input shape
        ov::element::Type,         // Model type
        std::string>;              // Device name

class BroadcastLayerTest : public testing::WithParamInterface<BroadcastParamsTuple>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BroadcastParamsTuple> &obj);

protected:
    void SetUp() override;
};
} //  namespace test
} //  namespace ov
