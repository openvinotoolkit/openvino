// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<InputShape>,  // inputShapes
        float,                    // score_threshold
        float,                    // nms_threshol
        float,                    // max_delta_log_wh
        int64_t,                  // num_classes
        int64_t,                  // post_nms_count
        size_t,                   // max_detections_per_image
        bool,                     // class_agnostic_box_regression
        std::vector<float>,       // deltas_weights
        ov::element::Type,        // Model type
        std::string               // Device name
> ExperimentalDetectronDetectionOutputTestParams;

class ExperimentalDetectronDetectionOutputLayerTest :
        public testing::WithParamInterface<ExperimentalDetectronDetectionOutputTestParams>,
        virtual public SubgraphBaseTest {
protected:
    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronDetectionOutputTestParams>& obj);
};
} // namespace test
} // namespace ov
