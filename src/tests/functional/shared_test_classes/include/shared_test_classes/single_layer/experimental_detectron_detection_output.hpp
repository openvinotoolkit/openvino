// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/utils/ov_helpers.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

typedef std::tuple<
        std::vector<InputShape>,                // inputShapes
        float,                                  // score_threshold
        float,                                  // nms_threshol
        float,                                  // max_delta_log_wh
        int64_t,                                // num_classes
        int64_t,                                // post_nms_count
        size_t,                                 // max_detections_per_image
        bool,                                   // class_agnostic_box_regression
        std::vector<float>,                     // deltas_weights
        ElementType,                            // Network precision
        std::string                             // Device name
> ExperimentalDetectronDetectionOutputTestParams;

class ExperimentalDetectronDetectionOutputLayerTest :
        public testing::WithParamInterface<ExperimentalDetectronDetectionOutputTestParams>,
        virtual public SubgraphBaseTest {
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronDetectionOutputTestParams>& obj);
};
} // namespace subgraph
} // namespace test
} // namespace ov
