// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {

using MemoryLayerTestParams = std::tuple<
        ov::test::utils::MemoryTransformation,   // Apply Memory transformation
        int64_t,                                 // iterationCount
        ov::Shape,                               // inputShape
        ov::element::Type,                       // modelType
        std::string                              // targetDevice
>;

class MemoryLayerTest : public testing::WithParamInterface<MemoryLayerTestParams>,
                   virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MemoryLayerTestParams> &obj);

protected:
    void SetUp() override;
    void infer() override;
    std::vector<ov::Tensor> calculate_refs() override;

    void CreateCommonFunc(ov::element::Type model_type, ov::Shape input_shape);
    void CreateTIFunc(ov::element::Type model_type, ov::Shape input_shape);
    void ApplyLowLatency(ov::test::utils::MemoryTransformation transformation);

    bool use_version_3 = false;
    int64_t iteration_count;
};

class MemoryV3LayerTest : public MemoryLayerTest {
protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
