// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/proposal.hpp"
#include "common_test_utils/test_constants.hpp"
#include "precision_utils.h"
#include "common_test_utils/test_constants.hpp"

// additional features for developer testing
//#define PROPOSAL_TESTS_LOGGING
const bool compareOutputScoresSoft = true; // do compare scores against reference with threshold
const float scoresThreshold = 0.1f; // threshold for soft scores comparison
const float ratioScores = 0.9f; // required ratio(matching_scores_count/count_optimized_scores) to assume
                                // that test is passed (for soft scores comparison)
const float ratioBoxes = 0.9f; // required ratio(matching_boxes_count/count_optimized_boxes) to assume
                                // that test is passed
const bool compareOutputScores = false; // do compare scores output against reference without threshold

using namespace InferenceEngine;
using namespace PrecisionUtils;

/**
* "IoU = intersection area / union area" of two boxes A, B
*   A, B: 4-dim array (x1, y1, x2, y2)
*/

namespace {

float check_iou(const float* A, const float* B, bool normalized) {
    if (A[0] > B[2] || A[1] > B[3] || A[2] < B[0] || A[3] < B[1]) {
        return 0.0f;
    } else {
        float additional = 0.f;
        if (!normalized)
            additional = 1.0f;

        // overlapped region (= box)
        const auto x1 = std::max(A[0], B[0]);
        const auto y1 = std::max(A[1], B[1]);
        const auto x2 = std::min(A[2], B[2]);
        const auto y2 = std::min(A[3], B[3]);

        // intersection area
        const auto width = std::max(0.0f, x2 - x1 + additional);
        const auto height = std::max(0.0f, y2 - y1 + additional);
        const auto area = width * height;

        // area of A, B
        const auto A_area = (A[2] - A[0] + additional) * (A[3] - A[1] + additional);
        const auto B_area = (B[2] - B[0] + additional) * (B[3] - B[1] + additional);

        // IoU
        return area / (A_area + B_area - area);
    }
}

std::size_t get_num_rois(const float* array, std::size_t size) {
    std::size_t count = 0;
    while (count < size && array[count] != -1.f)
        count += 5;
    return count / 5;
}

}// namespace
typedef std::vector<std::vector<int>> Graph;

class KunhsAlgorithm {
public:
    int findMaximumMatching(const Graph& graph, std::size_t second_part_size) {
        int matching_count = 0;
        std::vector<bool> used(graph.size(), false);
        std::vector<int> matching(second_part_size, -1); // -1 means there is no matching
        for (std::size_t vertex = 0; vertex < graph.size(); ++vertex) {
            if (try_kuhn(used, matching, graph, vertex)) {
                used.assign(graph.size(), false);
                ++matching_count;
            }
        }
        return matching_count;
    }

private:
    bool try_kuhn(std::vector<bool>& used,
                  std::vector<int>& matching,
                  const Graph& graph,
                  int cur_vertex) {
        if (used[cur_vertex]) return false;
        used[cur_vertex] = true;

        for (auto to : graph[cur_vertex]) {
            if (matching[to] == -1) {
                matching[to] = cur_vertex;
                return true;
            }
        }
        for (auto to : graph[cur_vertex]) {
            if (try_kuhn(used, matching, graph, matching[to])) {
                matching[to] = cur_vertex;
                return true;
            }
        }
        return false;
    }
};


class GraphComparison {
public:
    int calculateMatchingCount(const float* reference, std::size_t count_reference,
                               const float* optimized, std::size_t count_optimized,
                               const float* reference_scores, const float* output_scores,
                               bool normalized, bool withOutputScores) {
        validateData(reference, count_reference, optimized, count_optimized);

        //edges from vertexes of optimized boxes to vertexes of reference boxes
        Graph graph {make_graph(reference, count_reference, optimized, count_optimized,
                                reference_scores, output_scores,
                                normalized, withOutputScores)};

        KunhsAlgorithm method;
        auto matching_count = method.findMaximumMatching(graph, count_reference);

        return matching_count;
    }

    void set_threshold(float threshold) {
        _threshold = threshold;
    }

private:
    float _threshold;

    Graph make_graph(const float* reference, std::size_t count_reference,
                     const float* optimized, std::size_t count_optimized,
                     const float* reference_scores, const float* output_scores,
                     bool normalized, bool withOutputScores) {
        Graph graph(count_optimized);
        for (std::size_t i = 0; i < count_optimized; ++i) {
            float out_values[4]{};
            for (int k = 0; k < 4; k++)
                    out_values[k] = optimized[i * 5 + k + 1];
            for (std::size_t j = 0; j < count_reference; ++j) {
                float gt_values[4]{};
                for (int k = 0; k < 4; k++)
                    gt_values[k] = reference[j * 5 + k + 1];

                bool isScoresEqual = true;
                if (compareOutputScores && withOutputScores)
                    isScoresEqual = (output_scores[i] == reference_scores[j]);

                if (isScoresEqual && check_iou(out_values, gt_values, normalized) >= _threshold)
                        graph[i].push_back(j);
            }
        }
        return graph;
    }

    // if x1 > x2 or y1 > y2, then we can assume that the data is incorrect
    void validateData(const float* reference, std::size_t count_reference,
                      const float* optimized, std::size_t count_optimized) {
        for (std::size_t i = 0; i < count_reference; i++) {
            ASSERT_TRUE(reference[i * 5 + 1] <= reference[i * 5 + 3])
                   << "Incorrect data: x1 > x2 in reference boxes";
            ASSERT_TRUE(reference[i * 5 + 2] <= reference[i * 5 + 4])
                   << "Incorrect data: x1 > x2 in reference boxes";
        }

        for (std::size_t i = 0; i < count_optimized; i++) {
            ASSERT_TRUE(optimized[i * 5 + 1] <= optimized[i * 5 + 3])
                   << "Incorrect data: x1 > x2 in reference boxes";
            ASSERT_TRUE(optimized[i * 5 + 2] <= optimized[i * 5 + 4])
                   << "Incorrect data: x1 > x2 in reference boxes";
        }
    }
};

namespace LayerTestsDefinitions {
    const normalize_type normalize = true;
    const feat_stride_type feat_stride = 1;
    const box_size_scale_type box_size_scale = 2.0f;
    const box_coordinate_scale_type box_coordinate_scale = 2.0f;

class MyriadProposalLayerTest
        : public testing::WithParamInterface<proposalLayerTestParamsSet>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<proposalLayerTestParamsSet> obj) {
        proposalSpecificParams proposalParams;
        std::string targetDevice;
        std::tie(proposalParams, targetDevice) = obj.param;
        auto proposalPramString = SerializeProposalSpecificParams(proposalParams);

        std::ostringstream result;
        result << "targetDevice=" << targetDevice;

        return "MyriadProposalTests_" + proposalPramString + result.str();
    }

    static std::string SerializeProposalSpecificParams(proposalSpecificParams& params) {
        base_size_type base_size;
        pre_nms_topn_type pre_nms_topn;
        post_nms_topn_type post_nms_topn;
        nms_thresh_type nms_thresh;
        min_size_type min_size;
        ratio_type ratio;
        scale_type scale;
        clip_before_nms_type clip_before_nms;
        clip_after_nms_type clip_after_nms;
        framework_type framework;
        std::tie(base_size, pre_nms_topn,
                post_nms_topn,
                nms_thresh,
                min_size,
                ratio,
                scale,
                clip_before_nms,
                clip_after_nms,
                framework) = params;

        std::ostringstream result;
        result << "base_size=" << base_size << "_";
        result << "pre_nms_topn=" << pre_nms_topn << "_";
        result << "post_nms_topn=" << post_nms_topn << "_";
        result << "nms_thresh=" << nms_thresh << "_";
        result << "feat_stride=" << feat_stride << "_";
        result << "min_size=" << min_size << "_";
        result << "ratio = " << CommonTestUtils::vec2str(ratio) << "_";
        result << "scale = " << CommonTestUtils::vec2str(scale) << "_";
        result << "clip_before_nms=" << clip_before_nms << "_";
        result << "clip_after_nms=" << clip_after_nms << "_";
        result << "normalize=" << normalize << "_";
        result << "box_size_scale=" << box_size_scale << "_";
        result << "box_coordinate_scale=" << box_coordinate_scale << "_";
        result << "framework=" << framework << "_";

        return result.str();
    }

    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs) override {
        const float* reference_boxes = nullptr;
        const float* reference_scores = nullptr;
        const float* optimized_boxes = nullptr;
        const float* optimized_scores = nullptr;
        std::size_t size_reference = 0;
        std::size_t size_optimized = 0;
        for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
            const auto &expected = expectedOutputs[outputIndex].second;
            const auto &actual = actualOutputs[outputIndex];
            const auto &expectedBuffer = expected.data();

            auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
            IE_ASSERT(memory);
            const auto lockedMemory = memory->rmap();
            const auto actualBuffer = lockedMemory.as<const std::uint8_t *>();

            if (outputIndex == 0) {
                reference_boxes = reinterpret_cast<const float *>(expectedBuffer);
                optimized_boxes = reinterpret_cast<const float *>(actualBuffer);
                size_reference = expected.size() / 4;
                size_optimized = actual->byteSize() / 4;
            } else {
                reference_scores = reinterpret_cast<const float *>(expectedBuffer);
                optimized_scores = reinterpret_cast<const float *>(actualBuffer);
            }
        }

        const auto num_reference = get_num_rois(reference_boxes, size_reference);
        const auto num_optimized = get_num_rois(optimized_boxes, size_optimized);
        const bool withOutputScores = (reference_scores != nullptr);

        GraphComparison method;
        method.set_threshold(0.7);
        auto matching_count = method.calculateMatchingCount(reference_boxes, num_reference,
                                                            optimized_boxes, num_optimized,
                                                            reference_scores,
                                                            optimized_scores,
                                                            normalize,
                                                            withOutputScores);

        #ifdef PROPOSAL_TESTS_LOGGING
        std::cout << "Matching count: " << matching_count << "/" <<  num_optimized << std::endl;
        #endif

        float precision = static_cast<float>(matching_count) / num_optimized;
        bool res = (precision >= ratioBoxes);

        if (compareOutputScoresSoft && res) {
            int min_size = std::min(num_reference, num_optimized);
            int scores_count = 0;
            for (int i = 0; i < min_size; ++i) {
                if (fabs(reference_scores[i] - optimized_scores[i]) <= scoresThreshold)
                    scores_count++;
            }
            float score_ratio = static_cast<float>(scores_count) / num_optimized;
            res = (score_ratio >= ratioScores);
        }
        ASSERT_TRUE(res) << "PROPOSAL TEST failed with "
        << matching_count << " matched boxes of " << num_optimized << std::endl;
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        InferenceEngine::Blob::Ptr blobPtr;

        const std::string name = info.name();
        if (name == "a_scores") {
            blobPtr = FuncTestUtils::createAndFillBlobFloat(info.getTensorDesc(), 1, 0, 1000, 8234231);
        } else if (name == "b_boxes") {
            blobPtr = FuncTestUtils::createAndFillBlobFloatNormalDistribution(info.getTensorDesc(), 0.0f, 0.2f, 7235346);
        }

        return blobPtr;
    }
protected:
    void SetUp() override{
        proposalSpecificParams proposalParams;
        std::vector<float> img_info = {225.0f, 225.0f, 1.0f};

        std::tie(proposalParams, targetDevice) = this->GetParam();
        base_size_type base_size;
        pre_nms_topn_type pre_nms_topn;
        post_nms_topn_type post_nms_topn;
        nms_thresh_type nms_thresh;
        min_size_type min_size;
        ratio_type ratio;
        scale_type scale;
        clip_before_nms_type clip_before_nms;
        clip_after_nms_type clip_after_nms;
        framework_type framework;

        std::tie(base_size, pre_nms_topn,
                 post_nms_topn,
                 nms_thresh,
                 min_size,
                 ratio,
                 scale,
                 clip_before_nms,
                 clip_after_nms,
                 framework) = proposalParams;

        size_t bottom_w = base_size;
        size_t bottom_h = base_size;
        size_t num_anchors = ratio.size() * scale.size();

        std::vector<size_t> scoresShape = {1, 2 * num_anchors, bottom_h, bottom_w};
        std::vector<size_t> boxesShape  = {1, 4 * num_anchors, bottom_h, bottom_w};
        std::vector<size_t> imageInfoShape = {3};

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP16);
        // a_ and b_ are a workaround to solve alphabetic param sorting that destroys ordering
        auto params = ngraph::builder::makeParams(ngPrc, {{"a_scores", scoresShape}, {"b_boxes", boxesShape}});
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        auto proposal = std::dynamic_pointer_cast<ngraph::opset4::Proposal>(
                ngraph::builder::makeProposal(paramOuts[0], paramOuts[1], img_info, ngPrc,
                                              base_size,
                                              pre_nms_topn,
                                              post_nms_topn,
                                              nms_thresh,
                                              feat_stride,
                                              min_size,
                                              ratio,
                                              scale,
                                              clip_before_nms,
                                              clip_after_nms,
                                              normalize,
                                              box_size_scale,
                                              box_coordinate_scale,
                                              framework));

        ngraph::ResultVector results{
            std::make_shared<ngraph::opset1::Result>(proposal->output(0)),
            std::make_shared<ngraph::opset1::Result>(proposal->output(1))};
        function = std::make_shared<ngraph::Function>(results, params, "proposal");
}
};

} // namespace LayerTestsDefinitions
using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {
/* ============= Proposal ============= */
const std::vector<base_size_type> base_size_ = {16};
const std::vector<pre_nms_topn_type> pre_nms_topn_ = {600, 300, 100};
const std::vector<post_nms_topn_type> post_nms_topn_ = {100, 50};
const std::vector<nms_thresh_type> nms_thresh_ = {0.7f};
const std::vector<min_size_type> min_size_ = {1};
const std::vector<ratio_type> ratio_ = {{1.0f, 2.0f}};
const std::vector<scale_type> scale_ = {{1.2f, 1.5f}};
const std::vector<clip_before_nms_type> clip_before_nms_ = {false};
const std::vector<clip_after_nms_type> clip_after_nms_ = {false};

// empty string corresponds to Caffe framework
// Myriad plugin does not take this parameter; uses "" by default
const std::vector<framework_type> framework_ = {""};

const auto proposalParams = ::testing::Combine(
        ::testing::ValuesIn(base_size_),
        ::testing::ValuesIn(pre_nms_topn_),
        ::testing::ValuesIn(post_nms_topn_),
        ::testing::ValuesIn(nms_thresh_),
        ::testing::ValuesIn(min_size_),
        ::testing::ValuesIn(ratio_),
        ::testing::ValuesIn(scale_),
        ::testing::ValuesIn(clip_before_nms_),
        ::testing::ValuesIn(clip_after_nms_),
        ::testing::ValuesIn(framework_)
);

TEST_P(MyriadProposalLayerTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Proposal_tests, MyriadProposalLayerTest,
                        ::testing::Combine(
                                proposalParams,
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        MyriadProposalLayerTest::getTestCaseName
);

}  // namespace
