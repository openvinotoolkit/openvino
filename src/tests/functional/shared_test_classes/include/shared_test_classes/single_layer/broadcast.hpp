// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace LayerTestsDefinitions {

using BroadcastParamsTuple = typename std::tuple<
        InferenceEngine::SizeVector,       // target shape
        ngraph::AxisSet,                   // axes mapping
        ngraph::op::BroadcastType,         // broadcast mode
        InferenceEngine::SizeVector,       // Input shape
        InferenceEngine::Precision,        // Network precision
        std::string>;                      // Device name

class BroadcastLayerTestLegacy : public testing::WithParamInterface<BroadcastParamsTuple>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BroadcastParamsTuple> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions

namespace ov {
namespace test {

using BroadcastParamsTuple = typename std::tuple<
        std::vector<size_t>,       // target shape
        ov::AxisSet,               // axes mapping
        ov::op::BroadcastType,     // broadcast mode
        std::vector<size_t>,       // Input shape
        ov::element::Type,         // Network precision
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
