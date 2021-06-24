// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

namespace proposalTypes {

typedef size_t base_size_type;
typedef size_t pre_nms_topn_type;
typedef size_t post_nms_topn_type;
typedef float nms_thresh_type;
typedef size_t min_size_type;
typedef std::vector<float> ratio_type;
typedef std::vector<float> scale_type;
typedef bool clip_before_nms_type;
typedef bool clip_after_nms_type;
typedef bool normalize_type;
typedef size_t feat_stride_type;
typedef float box_size_scale_type;
typedef float box_coordinate_scale_type;
typedef std::string framework_type;

};  // namespace proposalTypes

using namespace proposalTypes;

typedef std::tuple<
        base_size_type,
        pre_nms_topn_type,
        post_nms_topn_type,
        nms_thresh_type,
        min_size_type,
        ratio_type,
        scale_type,
        clip_before_nms_type,
        clip_after_nms_type,
        framework_type> proposalSpecificParams;
typedef std::tuple<
        proposalSpecificParams,
        std::string> proposalLayerTestParamsSet;

class ProposalLayerTest
        : public testing::WithParamInterface<proposalLayerTestParamsSet>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<proposalLayerTestParamsSet> obj);
    static std::string SerializeProposalSpecificParams(proposalSpecificParams& params);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs) override;
    template <class T>
    void Compare(const T *expected, const T *actual, std::size_t size,
                        T threshold, const std::size_t output_index) {
        for (std::size_t i = 0; i < size; ++i) {
            const auto &ref = expected[i];
            const auto &res = actual[i];

            // verify until first -1 appears in the 1st output.
            if (output_index == 0 &&
                CommonTestUtils::ie_abs(ref - static_cast<T>(-1)) <= threshold) {
                // output0 shape = {x, 5}
                // output1 shape = {x}
                // setting the new_size for output1 verification
                num_selected_boxes = i / 5;
                return;
            }

            const auto absoluteDifference = CommonTestUtils::ie_abs(res - ref);
            if (absoluteDifference <= threshold) {
                continue;
            }

            const auto max = std::max(CommonTestUtils::ie_abs(res),
                                    CommonTestUtils::ie_abs(ref));
            float diff =
                static_cast<float>(absoluteDifference) / static_cast<float>(max);
            ASSERT_TRUE(max != 0 && (diff <= static_cast<float>(threshold)))
                << "Relative comparison of values expected: " << ref
                << " and actual: " << res << " at index " << i
                << " with threshold " << threshold << " failed";
        }
    }
protected:
    void SetUp() override;

private:
    size_t num_selected_boxes;
};

}  // namespace LayerTestsDefinitions
