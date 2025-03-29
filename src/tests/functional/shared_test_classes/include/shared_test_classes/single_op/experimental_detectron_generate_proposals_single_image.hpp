// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<InputShape>,                // Input shapes
        float,                                  // min_size: minimum box width & height
        float,                                  // nms_threshold: specifies NMS threshold
        int64_t,                                // post_nms_count: number of top-n proposals after NMS
        int64_t,                                // pre_nms_count: number of top-n proposals after NMS
        ElementType,                            // Model type
        std::string                             // Device name
> ExperimentalDetectronGenerateProposalsSingleImageTestParams;

class ExperimentalDetectronGenerateProposalsSingleImageLayerTest :
        public testing::WithParamInterface<ExperimentalDetectronGenerateProposalsSingleImageTestParams>,
        virtual public SubgraphBaseTest {
protected:
    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronGenerateProposalsSingleImageTestParams>& obj);
};
} // namespace test
} // namespace ov
