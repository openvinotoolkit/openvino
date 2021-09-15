// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace LayerTestsDefinitions {

struct StridedSliceSpecificParams {
    InferenceEngine::SizeVector inputShape;
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> strides;
    std::vector<int64_t> beginMask;
    std::vector<int64_t> endMask;
    std::vector<int64_t> newAxisMask;
    std::vector<int64_t> shrinkAxisMask;
    std::vector<int64_t> ellipsisAxisMask;
};

using StridedSliceParams = std::tuple<
        StridedSliceSpecificParams,
        InferenceEngine::Precision,        // Net precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        std::string,                       // Device name
        std::map<std::string, std::string> // Additional network configuration
>;

class StridedSliceLayerTest : public testing::WithParamInterface<StridedSliceParams>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StridedSliceParams> &obj);

protected:
    void SetUp() override;
};

struct Slice8SpecificParams {
        std::vector<int64_t> begin;
        std::vector<int64_t> end;
        std::vector<int64_t> strides;
        std::vector<int64_t> axes;
};

using Slice8Params = std::tuple<
        std::vector<ov::test::InputShape>,               // Parameters shapes
        Slice8SpecificParams,                            // Slice-8 specific parameters
        ov::test::ElementType,                           // Net precision
        std::string,                                     // Device name
        std::map<std::string, std::string>               // Additional network configuration
>;

class Slice8LayerTest : public testing::WithParamInterface<Slice8Params>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<Slice8Params> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
