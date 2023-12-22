// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        std::vector<int>,                  // Indices
        std::vector<size_t>,               // Indices shape
        int,                               // Gather axis
        std::vector<size_t>,               // Input shapes
        InferenceEngine::Precision,        // Network precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        std::string                        // Device name
> gatherParamsTuple;

class GatherLayerTestBase : virtual public LayerTestsUtils::LayerTestsCommon {
public:
    using ::testing::Test::SetUp;
protected:
    void SetUp(const gatherParamsTuple& params);
};

class GatherLayerTest : public testing::WithParamInterface<gatherParamsTuple>, public GatherLayerTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gatherParamsTuple> &obj);

protected:
    void SetUp() override;
};


typedef std::tuple<
    std::vector<size_t>,               // Input shapes
    std::vector<size_t>,               // Indices shape
    std::tuple<int, int>,              // Gather axis and batch
    InferenceEngine::Precision,        // Network precision
    InferenceEngine::Precision,        // Input precision
    InferenceEngine::Precision,        // Output precision
    InferenceEngine::Layout,           // Input layout
    InferenceEngine::Layout,           // Output layout
    std::string                        // Device name
> gather7ParamsTuple;

class Gather7LayerTest : public testing::WithParamInterface<gather7ParamsTuple>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple>& obj);

protected:
    void SetUp() override;
};

class Gather8LayerTest : public testing::WithParamInterface<gather7ParamsTuple>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple>& obj);

protected:
    void SetUp() override;
};

class Gather8IndiceScalarLayerTest : public testing::WithParamInterface<gather7ParamsTuple>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple>& obj);

protected:
    void SetUp() override;
};

typedef std::tuple<
    gather7ParamsTuple,
    std::vector<int>                   // indices data
> gather8withIndicesDataParamsTuple;

class Gather8withIndicesDataLayerTest : public testing::WithParamInterface<gather8withIndicesDataParamsTuple>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gather8withIndicesDataParamsTuple>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
