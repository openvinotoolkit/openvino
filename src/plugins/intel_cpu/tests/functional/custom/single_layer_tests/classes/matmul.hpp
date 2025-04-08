// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

enum class MatMulNodeType {
    MatMul,
    FullyConnected
};

struct ShapeRelatedParams {
    std::vector<InputShape> inputShapes;
    std::pair<bool, bool> transpose;
};

typedef std::tuple<ShapeRelatedParams,
                   ElementType,            // Network precision
                   ElementType,            // Input precision
                   ElementType,            // Output precision
                   utils::InputLayerType,  // Secondary input type
                   TargetDevice,           // Device name
                   ov::AnyMap              // Additional network configuration
                   >
    MatMulLayerTestParamsSet;

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
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

namespace MatMul {
   const std::vector<ElementType>& netPRCs();
   const std::vector<fusingSpecificParams>& matmulFusingParams();
   const std::vector<ov::AnyMap>& additionalConfig();
   const ov::AnyMap& emptyAdditionalConfig();
   const std::vector<CPUSpecificParams>& filterSpecificParams();
   const std::vector<ShapeRelatedParams>& IS2D_nightly();
   const std::vector<ShapeRelatedParams>& IS2D_smoke();
   const std::vector<ShapeRelatedParams>& IS3D_smoke();
}  // namespace MatMul
}  // namespace test
}  // namespace ov
