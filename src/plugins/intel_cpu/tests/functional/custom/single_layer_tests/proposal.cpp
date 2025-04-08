// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

namespace proposalTypes {

    typedef size_t base_size_type;
    typedef float box_coordinate_scale_type;
    typedef float box_size_scale_type;
    typedef bool clip_after_nms_type;
    typedef bool clip_before_nms_type;
    typedef size_t feat_stride_type;
    typedef std::string framework_type;
    typedef size_t min_size_type;
    typedef float nms_thresh_type;
    typedef bool normalize_type;
    typedef size_t post_nms_topn_type;
    typedef size_t pre_nms_topn_type;
    typedef std::vector<float> ratio_type;
    typedef std::vector<float> scale_type;

};  // namespace proposalTypes

using namespace proposalTypes;

using proposalSpecificParams = std::tuple<
    base_size_type,
    box_coordinate_scale_type,
    box_size_scale_type,
    clip_after_nms_type,
    clip_before_nms_type,
    feat_stride_type,
    framework_type,
    min_size_type,
    nms_thresh_type,
    normalize_type,
    post_nms_topn_type,
    pre_nms_topn_type,
    ratio_type,
    scale_type>;

using proposalLayerTestCPUParams = std::tuple<std::vector<InputShape>,  // Input shapes
                                              proposalSpecificParams,   // Node attributes
                                              ov::element::Type>;       // Network precision

class ProposalLayerCPUTest : public testing::WithParamInterface<proposalLayerTestCPUParams>,
                             public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<proposalLayerTestCPUParams> obj) {
        std::vector<InputShape> inputShapes;
        proposalSpecificParams proposalParams;
        ov::element::Type netPrecision;
        std::tie(inputShapes, proposalParams, netPrecision) = obj.param;

        base_size_type base_size;
        box_coordinate_scale_type box_coordinate_scale;
        box_size_scale_type box_size_scale;
        clip_after_nms_type clip_after_nms;
        clip_before_nms_type clip_before_nms;
        feat_stride_type feat_stride;
        framework_type framework;
        min_size_type min_size;
        nms_thresh_type nms_thresh;
        normalize_type normalize;
        post_nms_topn_type post_nms_topn;
        pre_nms_topn_type pre_nms_topn;
        ratio_type ratio;
        scale_type scale;
        std::tie(base_size, box_coordinate_scale, box_size_scale,
                 clip_after_nms, clip_before_nms, feat_stride,
                 framework, min_size, nms_thresh, normalize,
                 post_nms_topn, pre_nms_topn, ratio, scale) = proposalParams;

        std::ostringstream result;
        if (inputShapes.front().first.size() != 0) {
            result << "IS=(";
            for (const auto &shape : inputShapes) {
                result << ov::test::utils::partialShape2str({shape.first}) << "_";
            }
            result.seekp(-1, result.cur);
            result << ")_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "base_size=" << base_size << "_";
        result << "framework=" << framework << "_";
        result << "ratio=" << ov::test::utils::vec2str(ratio) << "_";
        result << "scale=" << ov::test::utils::vec2str(scale) << "_";
        result << "netPRC=" << netPrecision.to_string();
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        proposalSpecificParams proposalParams;
        ov::element::Type netPrecision;
        std::tie(inputShapes, proposalParams, netPrecision) = this->GetParam();

        base_size_type base_size;
        box_coordinate_scale_type box_coordinate_scale;
        box_size_scale_type box_size_scale;
        clip_after_nms_type clip_after_nms;
        clip_before_nms_type clip_before_nms;
        feat_stride_type feat_stride;
        framework_type framework;
        min_size_type min_size;
        nms_thresh_type nms_thresh;
        normalize_type normalize;
        post_nms_topn_type post_nms_topn;
        pre_nms_topn_type pre_nms_topn;
        ratio_type ratio;
        scale_type scale;
        std::tie(base_size, box_coordinate_scale, box_size_scale,
                 clip_after_nms, clip_before_nms, feat_stride,
                 framework, min_size, nms_thresh, normalize,
                 post_nms_topn, pre_nms_topn, ratio, scale) = proposalParams;

        selectedType = std::string("ref_any_") + netPrecision.to_string();
        init_input_shapes(inputShapes);

        ov::ParameterVector params;
        for (auto&& shape : {inputDynamicShapes[0], inputDynamicShapes[1], inputDynamicShapes[2]}) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));
        }

        ov::op::v0::Proposal::Attributes attrs;
        attrs.base_size = base_size;
        attrs.pre_nms_topn = pre_nms_topn;
        attrs.post_nms_topn = post_nms_topn;
        attrs.nms_thresh = nms_thresh;
        attrs.feat_stride = feat_stride;
        attrs.min_size = min_size;
        attrs.ratio = ratio;
        attrs.scale = scale;
        attrs.clip_before_nms = clip_before_nms;
        attrs.clip_after_nms = clip_after_nms;
        attrs.normalize = normalize;
        attrs.box_size_scale = box_size_scale;
        attrs.box_coordinate_scale = box_coordinate_scale;
        attrs.framework = framework;
        attrs.infer_probs = true;

        auto proposal = std::make_shared<ov::op::v4::Proposal>(params[0], params[1], params[2], attrs);

        ov::ResultVector results{
                std::make_shared<ov::op::v0::Result>(proposal->output(0)),
                std::make_shared<ov::op::v0::Result>(proposal->output(1))
        };

        function = std::make_shared<ov::Model>(results, params, "Proposal");
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 2) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);

                auto *dataPtr = tensor.data<float>();
                dataPtr[0] = dataPtr[1] = 225.0f;
                dataPtr[2] = 1.0f;
                if (tensor.get_size() == 4) dataPtr[3] = 1.0f;
            } else {
                    ov::test::utils::InputGenerateData in_data;
                    in_data.start_from = 0;
                    in_data.range = 10;
                    in_data.resolution = 1000;
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(ProposalLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Proposal");
}

namespace {

const std::vector<ov::element::Type> netPrecision = {
        ov::element::f32
};

std::vector<std::vector<ov::Shape>> staticInputShapesCase1 = {
        {{2, 30, 18, 22}, {2, 60, 18, 22}, {3}},
        {{1, 30, 18, 22}, {1, 60, 18, 22}, {3}},
        {{1, 30, 50, 80}, {1, 60, 50, 80}, {4}}
};

std::vector<std::vector<InputShape>> dynamicInputShapesCase1 = {
        {
                {
                        {{-1, 30, -1, -1}, {{2, 30, 75, 75}, {1, 30, 50, 80}, {3, 30, 80, 80}}},
                        {{-1, 60, -1, -1}, {{2, 60, 75, 75}, {1, 60, 50, 80}, {3, 60, 80, 80}}},
                        {{{3, 4}}, {{3}, {4}, {3}}}
                }
        },
        {
                {
                        {{-1, 30, {20, 40}, {10, 50}}, {{1, 30, 20, 10}, {1, 30, 20, 30}, {1, 30, 40, 35}}},
                        {{-1, 60, {20, 40}, {10, 50}}, {{1, 60, 20, 10}, {1, 60, 20, 30}, {1, 60, 40, 35}}},
                        {{3}, {{3}, {3}, {3}}}
                }
        }
};

const std::vector<base_size_type> base_size_1 = {16};
const std::vector<box_coordinate_scale_type> box_coordinate_scale_1 = {1};
const std::vector<box_size_scale_type> box_size_scale_1 = {1};
const std::vector<clip_after_nms_type> clip_after_nms_1 = {false};
const std::vector<clip_before_nms_type> clip_before_nms_1 = {true};
const std::vector<feat_stride_type> feat_stride_1 = {16};
const std::vector<framework_type> framework_1 = {""};
const std::vector<min_size_type> min_size_1 = {12};
const std::vector<nms_thresh_type> nms_thresh_1 = {0.699999988079071};
const std::vector<normalize_type> normalize_1 = {true};
const std::vector<post_nms_topn_type> post_nms_topn_1 = {300};
const std::vector<pre_nms_topn_type> pre_nms_topn_1 = {6000};
const std::vector<ratio_type> ratio_1 = {{0.5, 1.0, 2.0}};
const std::vector<scale_type> scale_1 = {{2.0, 4.0, 8.0, 16.0, 32.0}};

const auto proposalParamsCase1 = ::testing::Combine(
        ::testing::ValuesIn(base_size_1),
        ::testing::ValuesIn(box_coordinate_scale_1),
        ::testing::ValuesIn(box_size_scale_1),
        ::testing::ValuesIn(clip_after_nms_1),
        ::testing::ValuesIn(clip_before_nms_1),
        ::testing::ValuesIn(feat_stride_1),
        ::testing::ValuesIn(framework_1),
        ::testing::ValuesIn(min_size_1),
        ::testing::ValuesIn(nms_thresh_1),
        ::testing::ValuesIn(normalize_1),
        ::testing::ValuesIn(post_nms_topn_1),
        ::testing::ValuesIn(pre_nms_topn_1),
        ::testing::ValuesIn(ratio_1),
        ::testing::ValuesIn(scale_1)
);

INSTANTIATE_TEST_SUITE_P(smoke_Proposal_Static_Test_Case1, ProposalLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapesCase1)),
                                            proposalParamsCase1,
                                            ::testing::ValuesIn(netPrecision)),
                         ProposalLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Proposal_Dynamic_Test_Case1, ProposalLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(dynamicInputShapesCase1),
                                            proposalParamsCase1,
                                            ::testing::ValuesIn(netPrecision)),
                         ProposalLayerCPUTest::getTestCaseName);


std::vector<std::vector<ov::Shape>> staticInputShapesCase2 = {
        {{1, 24, 24, 30}, {1, 48, 24, 30}, {3}},
        {{1, 24, 38, 38}, {1, 48, 38, 38}, {4}}
};

std::vector<std::vector<InputShape>> dynamicInputShapesCase2 = {
        {
                {
                        {{1, 24, -1, -1}, {{1, 24, 38, 38}, {1, 24, 20, 12}, {1, 24, 15, 15}}},
                        {{1, 48, -1, -1}, {{1, 48, 38, 38}, {1, 48, 20, 12}, {1, 48, 15, 15}}},
                        {{{3, 4}}, {{4}, {3}, {4}}}
                }
        },
        {
                {
                        {{1, 24, {11, 38}, {11, 38}}, {{1, 24, 19, 11}, {1, 24, 15, 30}, {1, 24, 18, 17}}},
                        {{1, 48, {11, 38}, {11, 38}}, {{1, 48, 19, 11}, {1, 48, 15, 30}, {1, 48, 18, 17}}},
                        {{4}, {{4}, {4}, {4}}}
                }
        }
};

const std::vector<base_size_type> base_size_2 = {256};
const std::vector<box_coordinate_scale_type> box_coordinate_scale_2 = {10};
const std::vector<box_size_scale_type> box_size_scale_2 = {5};
const std::vector<clip_after_nms_type> clip_after_nms_2 = {false};
const std::vector<clip_before_nms_type> clip_before_nms_2 = {true};
const std::vector<feat_stride_type> feat_stride_2 = {16};
const std::vector<framework_type> framework_2 = {"tensorflow"};
const std::vector<min_size_type> min_size_2 = {1};
const std::vector<nms_thresh_type> nms_thresh_2 = {0.699999988079071};
const std::vector<normalize_type> normalize_2 = {true};
const std::vector<post_nms_topn_type> post_nms_topn_2 = {100};
const std::vector<pre_nms_topn_type> pre_nms_topn_2 = {2147483647};
const std::vector<ratio_type> ratio_2 = {{0.5, 1.0, 2.0}};
const std::vector<scale_type> scale_2 = {{0.25, 0.5, 1.0, 2.0}};

const auto proposalParamsCase2 = ::testing::Combine(
        ::testing::ValuesIn(base_size_2),
        ::testing::ValuesIn(box_coordinate_scale_2),
        ::testing::ValuesIn(box_size_scale_2),
        ::testing::ValuesIn(clip_after_nms_2),
        ::testing::ValuesIn(clip_before_nms_2),
        ::testing::ValuesIn(feat_stride_2),
        ::testing::ValuesIn(framework_2),
        ::testing::ValuesIn(min_size_2),
        ::testing::ValuesIn(nms_thresh_2),
        ::testing::ValuesIn(normalize_2),
        ::testing::ValuesIn(post_nms_topn_2),
        ::testing::ValuesIn(pre_nms_topn_2),
        ::testing::ValuesIn(ratio_2),
        ::testing::ValuesIn(scale_2)
);

INSTANTIATE_TEST_SUITE_P(smoke_Proposal_Static_Test_Case2, ProposalLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapesCase2)),
                                            proposalParamsCase2,
                                            ::testing::ValuesIn(netPrecision)),
                         ProposalLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Proposal_Dynamic_Test_Case2, ProposalLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(dynamicInputShapesCase2),
                                            proposalParamsCase2,
                                            ::testing::ValuesIn(netPrecision)),
                         ProposalLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
