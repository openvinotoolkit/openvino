// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

using Attrs = ov::op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes;
using ExperimentalROI = ov::op::v6::ExperimentalDetectronROIFeatureExtractor;

typedef std::tuple<
        std::vector<InputShape>,                // Input shapes
        int64_t,                                // Output size
        int64_t,                                // Sampling ratio
        std::vector<int64_t>,                   // Pyramid scales
        bool,                                   // Aligned
        ElementType,                            // Network precision
        std::string                             // Device name>;
> ExperimentalDetectronROIFeatureExtractorTestParams;

class ExperimentalDetectronROIFeatureExtractorLayerTest : public testing::WithParamInterface<ExperimentalDetectronROIFeatureExtractorTestParams>,
                                                          virtual public SubgraphBaseTest {
protected:
    void SetUp() override;

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronROIFeatureExtractorTestParams>& obj);

private:
    uint32_t mainPyramidScale;
};
} // namespace subgraph
} // namespace test
} // namespace ov
