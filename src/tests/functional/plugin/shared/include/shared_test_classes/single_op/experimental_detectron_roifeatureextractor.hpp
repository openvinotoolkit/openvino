// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<InputShape>,                // Input shapes
        int64_t,                                // Output size
        int64_t,                                // Sampling ratio
        std::vector<int64_t>,                   // Pyramid scales
        bool,                                   // Aligned
        ElementType,                            // Model type
        std::string                             // Device name>;
> ExperimentalDetectronROIFeatureExtractorTestParams;

class ExperimentalDetectronROIFeatureExtractorLayerTest : public testing::WithParamInterface<ExperimentalDetectronROIFeatureExtractorTestParams>,
                                                          virtual public SubgraphBaseTest {
protected:
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronROIFeatureExtractorTestParams>& obj);
};
} // namespace test
} // namespace ov
