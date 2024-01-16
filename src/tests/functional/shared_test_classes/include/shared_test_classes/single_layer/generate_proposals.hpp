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
        std::vector<InputShape>,                // Input shapes
        float,                                  // min_size: minimum box width & height
        float,                                  // nms_threshold: specifies NMS threshold
        int64_t,                                // post_nms_count: number of top-n proposals after NMS
        int64_t,                                // pre_nms_count: number of top-n proposals after NMS
        bool,                                   // normalized: specifies whether box is normalized or not
        std::pair<std::string, std::vector<ov::Tensor>>, // input tensors
        ElementType,                            // Network precision
        ElementType,                            // roi_num precision
        std::string                             // Device name>;
> GenerateProposalsTestParams;

class GenerateProposalsLayerTest :
        public testing::WithParamInterface<GenerateProposalsTestParams>,
        virtual public SubgraphBaseTest {
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<GenerateProposalsTestParams>& obj);
};
} // namespace subgraph
} // namespace test
} // namespace ov
