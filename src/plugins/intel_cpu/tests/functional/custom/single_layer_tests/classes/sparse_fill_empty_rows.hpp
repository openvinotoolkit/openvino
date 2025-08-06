// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"

namespace ov::test::SparseFillEmptyRows {

using SparseFillEmptyRowsSpecificParams = std::tuple<
        InputShape,                   // values shape
        InputShape,                   // indices shape
        std::vector<int64_t>,         // dense_shape values
        int64_t                       // default_value
>;

using SparseFillEmptyRowsLayerTestParams = std::tuple<
        SparseFillEmptyRowsSpecificParams,
        ElementType,                     // values precision
        ElementType,                     // indices precision
        ov::test::utils::InputLayerType, // secondary input type (constant or parameter)
        ov::test::TargetDevice
>;

using SparseFillEmptyRowsLayerCPUTestParamsSet = std::tuple<
        SparseFillEmptyRowsLayerTestParams,
        CPUTestUtils::CPUSpecificParams>;

class SparseFillEmptyRowsLayerCPUTest : public testing::WithParamInterface<SparseFillEmptyRowsLayerCPUTestParamsSet>,
                             public SubgraphBaseTest, public CPUTestUtils::CPUTestsBase {
public:
   static std::string getTestCaseName(const testing::TestParamInfo<SparseFillEmptyRowsLayerCPUTestParamsSet>& obj);
protected:
   void SetUp() override;
   void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

extern const std::vector<SparseFillEmptyRowsSpecificParams> SparseFillEmptyRowsParamsVector;
extern const std::vector<ov::test::utils::InputLayerType> secondaryInputTypes;
extern const std::vector<ElementType> indicesPrecisions;

}  // namespace ov::test::SparseFillEmptyRows
