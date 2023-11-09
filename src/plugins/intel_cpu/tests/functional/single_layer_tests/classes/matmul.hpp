// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/mat_mul.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "ov_models/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

enum class MatMulNodeType {
    MatMul,
    FullyConnected
};

struct ShapeRelatedParams {
    std::vector<InputShape> inputShapes;
    std::pair<bool, bool> transpose;
};

typedef std::tuple<
        ShapeRelatedParams,
        ElementType,        // Network precision
        ElementType,        // Input precision
        ElementType,        // Output precision
        ngraph::helpers::InputLayerType,   // Secondary input type
        TargetDevice,     // Device name
        std::map<std::string, std::string> // Additional network configuration
> MatMulLayerTestParamsSet;

using MatMulLayerCPUTestParamSet = std::tuple<MatMulLayerTestParamsSet,
                                              MatMulNodeType,
                                              fusingSpecificParams,
                                              CPUSpecificParams>;

class MatMulLayerCPUTest : public testing::WithParamInterface<MatMulLayerCPUTestParamSet>,
                           virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulLayerCPUTestParamSet>& obj);

protected:
    std::string cpuNodeType;

    template<typename T>
    void transpose(T& shape);

    void SetUp() override;
};

namespace MatMul {
   const std::vector<ElementType>& netPRCs();
   const std::vector<fusingSpecificParams>& matmulFusingParams();
   const std::vector<std::map<std::string, std::string>>& additionalConfig();
   const std::map<std::string, std::string>& emptyAdditionalConfig();
   const std::vector<CPUSpecificParams>& filterSpecificParams();
   const std::vector<ShapeRelatedParams>& IS2D_nightly();
   const std::vector<ShapeRelatedParams>& IS2D_smoke();
   const std::vector<ShapeRelatedParams>& IS3D_smoke();
} // namespace MatMul
} // namespace CPULayerTestsDefinitions
