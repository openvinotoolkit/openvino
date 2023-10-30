// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/utils/ov_helpers.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<
        std::vector<InputShape>,                // Input shapes
        float,                                  // min_size: minimum box width & height
        float,                                  // nms_threshold: specifies NMS threshold
        int64_t,                                // post_nms_count: number of top-n proposals after NMS
        int64_t,                                // pre_nms_count: number of top-n proposals after NMS
        bool,                                   // normalized: specifies whether box is normalized or not
        ElementType,                            // Model type
        ElementType,                            // roi_num precision
        std::string                             // Device name>;
> GenerateProposalsTestParams;

class GenerateProposalsLayerTest :
        public testing::WithParamInterface<GenerateProposalsTestParams>,
        virtual public SubgraphBaseTest {
protected:
    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GenerateProposalsTestParams>& obj);
};

} // namespace test
} // namespace ov
