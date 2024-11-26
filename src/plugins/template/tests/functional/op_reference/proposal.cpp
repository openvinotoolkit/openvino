// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/proposal.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ProposalV1Params {
    template <class IT>
    ProposalV1Params(const float iou_threshold,
                     const int min_bbox_size,
                     const int feature_stride,
                     const int pre_nms_topn,
                     const int post_nms_topn,
                     const size_t image_shape_num,
                     const size_t image_h,
                     const size_t image_w,
                     const size_t image_z,
                     const std::vector<float>& ratios,
                     const std::vector<float>& scales,
                     const size_t batch_size,
                     const size_t anchor_num,
                     const size_t feat_map_height,
                     const size_t feat_map_width,
                     const ov::element::Type& iType,
                     const std::vector<IT>& clsScoreValues,
                     const std::vector<IT>& bboxPredValues,
                     const std::vector<IT>& proposalValues,
                     const std::string& test_name = "")
        : inType(iType),
          outType(iType),
          clsScoreData(CreateTensor(iType, clsScoreValues)),
          bboxPredData(CreateTensor(iType, bboxPredValues)),
          refProposalData(CreateTensor(Shape{batch_size * post_nms_topn, 5}, iType, proposalValues)),
          testcaseName(test_name) {
        clsScoreShape = Shape{batch_size, anchor_num * 2, feat_map_height, feat_map_width};
        bboxPredShape = Shape{batch_size, anchor_num * 4, feat_map_height, feat_map_width};
        imageShapeShape = Shape{image_shape_num};

        attrs.base_size = min_bbox_size;
        attrs.min_size = min_bbox_size;
        attrs.pre_nms_topn = pre_nms_topn;
        attrs.post_nms_topn = post_nms_topn;
        attrs.nms_thresh = iou_threshold;
        attrs.feat_stride = feature_stride;
        attrs.min_size = min_bbox_size;
        attrs.ratio = ratios;
        attrs.scale = scales;
        attrs.clip_before_nms = true;
        attrs.clip_after_nms = false;
        attrs.normalize = false;
        attrs.box_size_scale = 1.0f;
        attrs.box_coordinate_scale = 1.0f;
        attrs.framework = "";
        attrs.infer_probs = false;

        std::vector<IT> inputShapeValues;
        inputShapeValues.push_back(static_cast<IT>(image_h));
        inputShapeValues.push_back(static_cast<IT>(image_w));
        inputShapeValues.push_back(static_cast<IT>(image_z));
        imageShapeData = CreateTensor(iType, inputShapeValues);
    }

    ov::op::v0::Proposal::Attributes attrs;
    ov::PartialShape clsScoreShape;
    ov::PartialShape bboxPredShape;
    ov::PartialShape imageShapeShape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor clsScoreData;
    ov::Tensor bboxPredData;
    ov::Tensor imageShapeData;
    ov::Tensor refProposalData;
    std::string testcaseName;
};

struct ProposalV4Params {
    template <class IT>
    ProposalV4Params(const float iou_threshold,
                     const int min_bbox_size,
                     const int feature_stride,
                     const int pre_nms_topn,
                     const int post_nms_topn,
                     const std::vector<float>& ratios,
                     const std::vector<float>& scales,
                     const size_t batch_size,
                     const size_t anchor_num,
                     const size_t feat_map_height,
                     const size_t feat_map_width,
                     const ov::element::Type& iType,
                     const std::vector<IT>& clsScoreValues,
                     const std::vector<IT>& bboxPredValues,
                     const std::vector<IT>& inputInfoValues,
                     const std::vector<IT>& proposalValues,
                     const std::vector<IT>& probsValues,
                     const std::string& framework,
                     const std::string& test_name = "")
        : inType(iType),
          outType(iType),
          clsScoreData(CreateTensor(iType, clsScoreValues)),
          bboxPredData(CreateTensor(iType, bboxPredValues)),
          imageInfoData(CreateTensor(iType, inputInfoValues)),
          refProposalData(CreateTensor(Shape{batch_size * post_nms_topn, 5}, iType, proposalValues)),
          refProbsData(CreateTensor(Shape{batch_size * post_nms_topn}, iType, probsValues)),
          testcaseName(test_name) {
        clsScoreShape = Shape{batch_size, anchor_num * 2, feat_map_height, feat_map_width};
        bboxPredShape = Shape{batch_size, anchor_num * 4, feat_map_height, feat_map_width};
        imageInfoShape = Shape{inputInfoValues.size()};

        attrs.base_size = min_bbox_size;
        attrs.min_size = min_bbox_size;
        attrs.pre_nms_topn = pre_nms_topn;
        attrs.post_nms_topn = post_nms_topn;
        attrs.nms_thresh = iou_threshold;
        attrs.feat_stride = feature_stride;
        attrs.min_size = min_bbox_size;
        attrs.ratio = ratios;
        attrs.scale = scales;
        attrs.clip_before_nms = true;
        attrs.clip_after_nms = false;
        attrs.normalize = false;
        attrs.box_size_scale = 1.0f;
        attrs.box_coordinate_scale = 1.0f;
        attrs.framework = framework;
        attrs.infer_probs = true;
    }

    ov::op::v4::Proposal::Attributes attrs;
    ov::PartialShape clsScoreShape;
    ov::PartialShape bboxPredShape;
    ov::PartialShape imageInfoShape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor clsScoreData;
    ov::Tensor bboxPredData;
    ov::Tensor imageInfoData;
    ov::Tensor refProposalData;
    ov::Tensor refProbsData;
    std::string testcaseName;
};

class ReferenceProposalV1LayerTest : public testing::TestWithParam<ProposalV1Params>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.clsScoreData, params.bboxPredData, params.imageShapeData};
        refOutData = {params.refProposalData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ProposalV1Params>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "clsScoreShape=" << param.clsScoreShape << "_";
        result << "bboxPredShape=" << param.bboxPredShape << "_";
        result << "imageShapeShape=" << param.imageShapeShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        if (!param.testcaseName.empty())
            result << "_" << param.testcaseName;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ProposalV1Params& params) {
        const auto class_probs_param = std::make_shared<op::v0::Parameter>(params.inType, params.clsScoreShape);
        const auto bbox_deltas_param = std::make_shared<op::v0::Parameter>(params.inType, params.bboxPredShape);
        const auto image_shape_param = std::make_shared<op::v0::Parameter>(params.inType, params.imageShapeShape);
        const auto Proposal =
            std::make_shared<op::v0::Proposal>(class_probs_param, bbox_deltas_param, image_shape_param, params.attrs);
        return std::make_shared<ov::Model>(NodeVector{Proposal},
                                           ParameterVector{class_probs_param, bbox_deltas_param, image_shape_param});
    }
};

class ReferenceProposalV4LayerTest : public testing::TestWithParam<ProposalV4Params>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.clsScoreData, params.bboxPredData, params.imageInfoData};
        refOutData = {params.refProposalData, params.refProbsData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ProposalV4Params>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "clsScoreShape=" << param.clsScoreShape << "_";
        result << "bboxPredShape=" << param.bboxPredShape << "_";
        result << "imageInfoShape=" << param.imageInfoShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        if (!param.attrs.framework.empty())
            result << "_" << param.attrs.framework;
        if (!param.testcaseName.empty())
            result << "_" << param.testcaseName;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ProposalV4Params& params) {
        const auto class_probs_param = std::make_shared<op::v0::Parameter>(params.inType, params.clsScoreShape);
        const auto bbox_deltas_param = std::make_shared<op::v0::Parameter>(params.inType, params.bboxPredShape);
        const auto image_shape_param = std::make_shared<op::v0::Parameter>(params.inType, params.imageInfoShape);
        const auto Proposal =
            std::make_shared<op::v4::Proposal>(class_probs_param, bbox_deltas_param, image_shape_param, params.attrs);
        return std::make_shared<ov::Model>(Proposal->outputs(),
                                           ParameterVector{class_probs_param, bbox_deltas_param, image_shape_param});
    }
};

TEST_P(ReferenceProposalV1LayerTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceProposalV4LayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ProposalV1Params> generateProposalV1Params() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ProposalV1Params> proposalV1Params{
        ProposalV1Params(
            0.7f,     // iou_threshold
            16,       // min_nnox_size
            16,       // feature_stride
            6000,     // pre_nms_topn
            10,       // post_nms_topn
            3,        // image_shape_num
            210,      // image_h
            350,      // image_w
            1,        // image_z
            {0.5f},   // ratios
            {32.0f},  // scales
            1,        // batch_size
            1,        // anchor_num
            10,       // feat_map_height
            10,       // feat_map_width
            IN_ET,
            std::vector<T>{
                0.000240f, 0.003802f, 0.111432f, 0.000503f, 0.007887f, 0.144701f, 0.399074f, 0.004680f,  // 0
                0.139741f, 0.002386f, 0.030003f, 0.276552f, 0.000267f, 0.022971f, 0.287953f, 0.050235f,  // 8
                0.002580f, 0.206311f, 0.000146f, 0.009656f, 0.175462f, 0.000147f, 0.014718f, 0.272348f,  // 16
                0.065199f, 0.003286f, 0.185335f, 0.003720f, 0.025932f, 0.251401f, 0.001465f, 0.090447f,  // 24
                0.488469f, 0.092259f, 0.019306f, 0.379091f, 0.005311f, 0.010369f, 0.087615f, 0.042003f,  // 32
                0.073871f, 0.416763f, 0.044282f, 0.069776f, 0.313032f, 0.000457f, 0.017346f, 0.089762f,  // 40
                0.000820f, 0.103986f, 0.367993f, 0.026315f, 0.035701f, 0.299252f, 0.000135f, 0.017825f,  // 48
                0.150119f, 0.000076f, 0.050511f, 0.269601f, 0.026680f, 0.003541f, 0.189765f, 0.000051f,  // 56
                0.004315f, 0.193150f, 0.000032f, 0.007254f, 0.185557f, 0.051526f, 0.000657f, 0.117579f,  // 64
                0.000115f, 0.010179f, 0.293187f, 0.000025f, 0.006505f, 0.175345f, 0.032587f, 0.000469f,  // 72
                0.098443f, 0.000121f, 0.009600f, 0.322782f, 0.000032f, 0.004543f, 0.166860f, 0.044911f,  // 80
                0.000187f, 0.102691f, 0.000242f, 0.005502f, 0.107865f, 0.000191f, 0.005336f, 0.086893f,  // 88
                0.078422f, 0.000345f, 0.079096f, 0.000281f, 0.016388f, 0.214072f, 0.000107f, 0.012027f,  // 96
                0.192754f, 0.049531f, 0.000386f, 0.149893f, 0.000374f, 0.016965f, 0.204781f, 0.000163f,  // 104
                0.016272f, 0.215277f, 0.032298f, 0.000857f, 0.133426f, 0.000614f, 0.020215f, 0.165789f,  // 112
                0.000225f, 0.036951f, 0.262195f, 0.087675f, 0.004596f, 0.147764f, 0.000219f, 0.010502f,  // 120
                0.163394f, 0.000152f, 0.023116f, 0.241702f, 0.081800f, 0.002197f, 0.146637f, 0.000193f,  // 128
                0.012017f, 0.133497f, 0.000375f, 0.028605f, 0.309179f, 0.065962f, 0.005508f, 0.155530f,  // 136
                0.000186f, 0.004540f, 0.079319f, 0.000799f, 0.031003f, 0.303045f, 0.051473f, 0.017770f,  // 144
                0.206188f, 0.000202f, 0.004291f, 0.061095f, 0.001109f, 0.018094f, 0.156639f, 0.026062f,  // 152
                0.005270f, 0.148651f, 0.000026f, 0.007300f, 0.096013f, 0.000383f, 0.022134f, 0.129511f,  // 160
                0.080882f, 0.003416f, 0.129922f, 0.000037f, 0.010040f, 0.130007f, 0.000116f, 0.014904f,  // 168
                0.171423f, 0.082893f, 0.000921f, 0.154976f, 0.000142f, 0.016552f, 0.209696f, 0.000227f,  // 176
                0.022418f, 0.228501f, 0.111712f, 0.001987f, 0.158164f, 0.001200f, 0.027049f, 0.308222f,  // 184
                0.001366f, 0.038146f, 0.287945f, 0.072526f, 0.016064f, 0.257895f, 0.000595f, 0.016962f,  // 192
            },
            std::vector<T>{
                0.006756f,  -0.055635f, 0.030843f,  0.007482f,  0.009056f,  -0.041824f, 0.119722f,  0.168988f,
                0.002822f,  0.039733f,  0.109005f,  0.245152f,  -0.013196f, -0.018222f, -0.170122f, -0.374904f,
                -0.005455f, -0.034059f, -0.006787f, 0.072005f,  -0.017933f, -0.007358f, 0.034149f,  0.123846f,
                0.128319f,  0.016107f,  -0.615487f, -1.235094f, -0.024253f, -0.019406f, 0.134142f,  0.157853f,
                -0.021119f, 0.007383f,  0.089365f,  0.092854f,  0.062491f,  0.002366f,  0.122464f,  -0.003326f,
                0.015468f,  -0.034088f, 0.079009f,  0.075483f,  0.011972f,  0.042427f,  0.106865f,  0.158754f,
                0.071211f,  -0.034009f, 0.007985f,  -0.441477f, 0.009046f,  -0.028515f, 0.095372f,  0.119598f,
                -0.007553f, -0.0072f,   0.105072f,  0.084314f,  0.23268f,   -0.02906f,  -0.408454f, -1.13439f,
                0.016202f,  -0.037859f, 0.130873f,  0.129652f,  0.002064f,  -0.011969f, 0.171623f,  0.050218f,
                0.113831f,  0.028922f,  0.017785f,  0.059708f,  0.037658f,  -0.011245f, 0.097197f,  0.137491f,
                0.024218f,  0.04739f,   0.091978f,  0.217333f,  0.088418f,  -0.004662f, -0.095168f, -0.397928f,
                0.02639f,   -0.008501f, 0.068487f,  0.108465f,  0.020069f,  0.018829f,  0.040206f,  0.068473f,
                0.226458f,  -0.072871f, -0.672384f, -1.447558f, 0.039598f,  0.017471f,  0.187288f,  0.08409f,
                0.017152f,  -0.00516f,  0.183419f,  0.068469f,  0.063944f,  0.160725f,  -0.022493f, -0.132291f,
                0.010542f,  0.036318f,  0.074042f,  -0.013323f, 0.00808f,   0.060365f,  0.120566f,  0.21866f,
                0.046324f,  0.088741f,  0.029469f,  -0.517183f, 0.00917f,   0.011915f,  0.053674f,  0.140168f,
                0.0033f,    0.022759f,  -0.006196f, 0.063839f,  0.083726f,  -0.088385f, -0.57208f,  -1.454211f,
                0.020655f,  0.010788f,  0.134951f,  0.109709f,  0.015445f,  -0.015363f, 0.109153f,  0.051209f,
                0.024297f,  0.139126f,  -0.12358f,  -0.127979f, 0.004587f,  0.004751f,  0.047292f,  0.027066f,
                0.011003f,  0.069887f,  0.117052f,  0.267419f,  0.039306f,  0.077584f,  0.02579f,   -0.496149f,
                -0.005569f, 0.015494f,  -0.011662f, 0.105549f,  -0.007015f, 0.031984f,  -0.075742f, 0.0852f,
                0.023886f,  -0.053107f, -0.325533f, -1.329066f, 0.004688f,  0.034501f,  0.089317f,  0.042463f,
                0.004212f,  -0.015128f, 0.00892f,   0.028266f,  0.009997f,  0.157822f,  0.020116f,  -0.142337f,
                0.008199f,  0.046564f,  0.083014f,  0.046307f,  0.006771f,  0.084997f,  0.141935f,  0.228339f,
                -0.020308f, 0.077745f,  -0.018319f, -0.522311f, 0.010432f,  0.024641f,  0.020571f,  0.097148f,
                0.002064f,  0.035053f,  -0.121995f, 0.012222f,  -0.030779f, 0.100481f,  -0.331737f, -1.257669f,
                -0.013079f, 0.021227f,  0.159949f,  0.120097f,  0.005765f,  -0.012335f, -0.005268f, 0.042067f,
                -0.043972f, 0.102556f,  0.180494f,  -0.084721f, -0.011962f, 0.031302f,  0.112511f,  0.027557f,
                -0.002085f, 0.082978f,  0.149409f,  0.195091f,  -0.033731f, 0.019861f,  -0.064047f, -0.471328f,
                -0.004093f, 0.016803f,  0.044635f,  0.058912f,  -0.018735f, 0.035536f,  -0.050373f, -0.002794f,
                -0.086705f, 0.038435f,  -0.301466f, -1.071246f, -0.028247f, 0.018984f,  0.254702f,  0.141142f,
                -0.017522f, 0.014843f,  0.079391f,  0.079662f,  -0.051204f, 0.048419f,  0.235604f,  -0.185797f,
                -0.019569f, 0.02678f,   0.162507f,  0.046435f,  -0.004606f, 0.08806f,   0.18634f,   0.193957f,
                -0.024333f, -0.01298f,  -0.17977f,  -0.65881f,  -0.003778f, 0.007418f,  0.065439f,  0.104549f,
                -0.027706f, 0.03301f,   0.057492f,  0.032019f,  -0.135337f, 0.000269f,  -0.250203f, -1.181688f,
                -0.027022f, -0.006755f, 0.206848f,  0.129268f,  -0.003529f, 0.013445f,  0.181484f,  0.139955f,
                -0.036587f, 0.065824f,  0.288751f,  -0.110813f, -0.015578f, 0.044818f,  0.17756f,   0.006914f,
                0.002329f,  0.068982f,  0.189079f,  0.184253f,  0.00301f,   -0.039168f, -0.010855f, -0.393254f,
                0.000028f,  0.001906f,  0.07217f,   0.063305f,  -0.026144f, 0.028842f,  0.139149f,  0.023377f,
                0.023362f,  0.023559f,  -0.145386f, -0.863572f, -0.015749f, -0.021364f, 0.172571f,  0.078393f,
                -0.037253f, 0.014978f,  0.221502f,  0.189111f,  -0.048956f, 0.085409f,  0.325399f,  -0.058294f,
                -0.028495f, 0.021663f,  0.19392f,   0.02706f,   0.006908f,  0.065751f,  0.176395f,  0.138375f,
                0.012418f,  -0.031228f, -0.008762f, -0.427345f, -0.013677f, -0.002429f, 0.069655f,  0.019505f,
                -0.036763f, 0.022528f,  0.201062f,  0.022205f,  0.024528f,  0.06241f,   -0.076237f, -0.840695f,
                -0.007268f, -0.027865f, 0.211056f,  0.074744f,  -0.053563f, 0.006863f,  0.301432f,  0.192879f,
                -0.021944f, 0.100535f,  0.19031f,   -0.133746f, -0.006151f, 0.023944f,  0.13561f,   -0.03259f,
                0.000618f,  0.063736f,  0.180904f,  0.12393f,   0.001275f,  -0.0306f,   -0.032822f, -0.496515f,
                0.009757f,  0.014602f,  0.004532f,  -0.039969f, -0.015984f, 0.047726f,  0.099865f,  0.003163f,
                0.026623f,  0.117951f,  -0.076234f, -0.811997f, 0.01301f,   0.020042f,  0.173756f,  -0.036191f,
                -0.068887f, 0.0229f,    0.245465f,  0.214282f,  -0.011054f, 0.132813f,  0.241014f,  -0.148763f,
            },
            std::vector<T>{
                0.000000f, 0.000000f,   0.000000f,  349.000000f, 209.000000f,  // 0
                0.000000f, 0.000000f,   0.000000f,  237.625443f, 209.000000f,  // 5
                0.000000f, 140.305511f, 0.000000f,  349.000000f, 209.000000f,  // 10
                0.000000f, 0.000000f,   0.000000f,  349.000000f, 65.359818f,   // 15
                0.000000f, 0.000000f,   0.000000f,  349.000000f, 130.324097f,  // 20
                0.000000f, 0.000000f,   15.562508f, 97.587891f,  181.224182f,  // 25
                0.000000f, 0.000000f,   68.539543f, 250.406708f, 209.000000f,  // 30
                0.000000f, 0.000000f,   0.000000f,  195.881531f, 99.841385f,   // 35
                0.000000f, 0.000000f,   0.000000f,  78.303986f,  209.000000f,  // 40
                0.000000f, 0.000000f,   0.000000f,  0.000000f,   209.000000f,  // 45
            }),
    };
    return proposalV1Params;
}

template <element::Type_t IN_ET>
std::vector<ProposalV4Params> generateProposalV4Params() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ProposalV4Params> proposalV4Params{
        ProposalV4Params{
            0.7f,     // iou_threshold
            16,       // min_bbox_size
            16,       // feature_stride
            6000,     // pre_nms_topn
            10,       // post_nms_topn
            {0.5f},   // ratios
            {32.0f},  // scales
            1,        // batch_size
            1,        // anchor_num
            10,       // feat_map_height
            10,       // feat_map_width
            IN_ET,
            std::vector<T>{
                0.000240f, 0.003802f, 0.111432f, 0.000503f, 0.007887f, 0.144701f, 0.399074f, 0.004680f,  // 0
                0.139741f, 0.002386f, 0.030003f, 0.276552f, 0.000267f, 0.022971f, 0.287953f, 0.050235f,  // 8
                0.002580f, 0.206311f, 0.000146f, 0.009656f, 0.175462f, 0.000147f, 0.014718f, 0.272348f,  // 16
                0.065199f, 0.003286f, 0.185335f, 0.003720f, 0.025932f, 0.251401f, 0.001465f, 0.090447f,  // 24
                0.488469f, 0.092259f, 0.019306f, 0.379091f, 0.005311f, 0.010369f, 0.087615f, 0.042003f,  // 32
                0.073871f, 0.416763f, 0.044282f, 0.069776f, 0.313032f, 0.000457f, 0.017346f, 0.089762f,  // 40
                0.000820f, 0.103986f, 0.367993f, 0.026315f, 0.035701f, 0.299252f, 0.000135f, 0.017825f,  // 48
                0.150119f, 0.000076f, 0.050511f, 0.269601f, 0.026680f, 0.003541f, 0.189765f, 0.000051f,  // 56
                0.004315f, 0.193150f, 0.000032f, 0.007254f, 0.185557f, 0.051526f, 0.000657f, 0.117579f,  // 64
                0.000115f, 0.010179f, 0.293187f, 0.000025f, 0.006505f, 0.175345f, 0.032587f, 0.000469f,  // 72
                0.098443f, 0.000121f, 0.009600f, 0.322782f, 0.000032f, 0.004543f, 0.166860f, 0.044911f,  // 80
                0.000187f, 0.102691f, 0.000242f, 0.005502f, 0.107865f, 0.000191f, 0.005336f, 0.086893f,  // 88
                0.078422f, 0.000345f, 0.079096f, 0.000281f, 0.016388f, 0.214072f, 0.000107f, 0.012027f,  // 96
                0.192754f, 0.049531f, 0.000386f, 0.149893f, 0.000374f, 0.016965f, 0.204781f, 0.000163f,  // 104
                0.016272f, 0.215277f, 0.032298f, 0.000857f, 0.133426f, 0.000614f, 0.020215f, 0.165789f,  // 112
                0.000225f, 0.036951f, 0.262195f, 0.087675f, 0.004596f, 0.147764f, 0.000219f, 0.010502f,  // 120
                0.163394f, 0.000152f, 0.023116f, 0.241702f, 0.081800f, 0.002197f, 0.146637f, 0.000193f,  // 128
                0.012017f, 0.133497f, 0.000375f, 0.028605f, 0.309179f, 0.065962f, 0.005508f, 0.155530f,  // 136
                0.000186f, 0.004540f, 0.079319f, 0.000799f, 0.031003f, 0.303045f, 0.051473f, 0.017770f,  // 144
                0.206188f, 0.000202f, 0.004291f, 0.061095f, 0.001109f, 0.018094f, 0.156639f, 0.026062f,  // 152
                0.005270f, 0.148651f, 0.000026f, 0.007300f, 0.096013f, 0.000383f, 0.022134f, 0.129511f,  // 160
                0.080882f, 0.003416f, 0.129922f, 0.000037f, 0.010040f, 0.130007f, 0.000116f, 0.014904f,  // 168
                0.171423f, 0.082893f, 0.000921f, 0.154976f, 0.000142f, 0.016552f, 0.209696f, 0.000227f,  // 176
                0.022418f, 0.228501f, 0.111712f, 0.001987f, 0.158164f, 0.001200f, 0.027049f, 0.308222f,  // 184
                0.001366f, 0.038146f, 0.287945f, 0.072526f, 0.016064f, 0.257895f, 0.000595f, 0.016962f,  // 192
            },
            std::vector<T>{
                0.006756f,  -0.055635f, 0.030843f,  0.007482f,  0.009056f,  -0.041824f, 0.119722f,  0.168988f,
                0.002822f,  0.039733f,  0.109005f,  0.245152f,  -0.013196f, -0.018222f, -0.170122f, -0.374904f,
                -0.005455f, -0.034059f, -0.006787f, 0.072005f,  -0.017933f, -0.007358f, 0.034149f,  0.123846f,
                0.128319f,  0.016107f,  -0.615487f, -1.235094f, -0.024253f, -0.019406f, 0.134142f,  0.157853f,
                -0.021119f, 0.007383f,  0.089365f,  0.092854f,  0.062491f,  0.002366f,  0.122464f,  -0.003326f,
                0.015468f,  -0.034088f, 0.079009f,  0.075483f,  0.011972f,  0.042427f,  0.106865f,  0.158754f,
                0.071211f,  -0.034009f, 0.007985f,  -0.441477f, 0.009046f,  -0.028515f, 0.095372f,  0.119598f,
                -0.007553f, -0.0072f,   0.105072f,  0.084314f,  0.23268f,   -0.02906f,  -0.408454f, -1.13439f,
                0.016202f,  -0.037859f, 0.130873f,  0.129652f,  0.002064f,  -0.011969f, 0.171623f,  0.050218f,
                0.113831f,  0.028922f,  0.017785f,  0.059708f,  0.037658f,  -0.011245f, 0.097197f,  0.137491f,
                0.024218f,  0.04739f,   0.091978f,  0.217333f,  0.088418f,  -0.004662f, -0.095168f, -0.397928f,
                0.02639f,   -0.008501f, 0.068487f,  0.108465f,  0.020069f,  0.018829f,  0.040206f,  0.068473f,
                0.226458f,  -0.072871f, -0.672384f, -1.447558f, 0.039598f,  0.017471f,  0.187288f,  0.08409f,
                0.017152f,  -0.00516f,  0.183419f,  0.068469f,  0.063944f,  0.160725f,  -0.022493f, -0.132291f,
                0.010542f,  0.036318f,  0.074042f,  -0.013323f, 0.00808f,   0.060365f,  0.120566f,  0.21866f,
                0.046324f,  0.088741f,  0.029469f,  -0.517183f, 0.00917f,   0.011915f,  0.053674f,  0.140168f,
                0.0033f,    0.022759f,  -0.006196f, 0.063839f,  0.083726f,  -0.088385f, -0.57208f,  -1.454211f,
                0.020655f,  0.010788f,  0.134951f,  0.109709f,  0.015445f,  -0.015363f, 0.109153f,  0.051209f,
                0.024297f,  0.139126f,  -0.12358f,  -0.127979f, 0.004587f,  0.004751f,  0.047292f,  0.027066f,
                0.011003f,  0.069887f,  0.117052f,  0.267419f,  0.039306f,  0.077584f,  0.02579f,   -0.496149f,
                -0.005569f, 0.015494f,  -0.011662f, 0.105549f,  -0.007015f, 0.031984f,  -0.075742f, 0.0852f,
                0.023886f,  -0.053107f, -0.325533f, -1.329066f, 0.004688f,  0.034501f,  0.089317f,  0.042463f,
                0.004212f,  -0.015128f, 0.00892f,   0.028266f,  0.009997f,  0.157822f,  0.020116f,  -0.142337f,
                0.008199f,  0.046564f,  0.083014f,  0.046307f,  0.006771f,  0.084997f,  0.141935f,  0.228339f,
                -0.020308f, 0.077745f,  -0.018319f, -0.522311f, 0.010432f,  0.024641f,  0.020571f,  0.097148f,
                0.002064f,  0.035053f,  -0.121995f, 0.012222f,  -0.030779f, 0.100481f,  -0.331737f, -1.257669f,
                -0.013079f, 0.021227f,  0.159949f,  0.120097f,  0.005765f,  -0.012335f, -0.005268f, 0.042067f,
                -0.043972f, 0.102556f,  0.180494f,  -0.084721f, -0.011962f, 0.031302f,  0.112511f,  0.027557f,
                -0.002085f, 0.082978f,  0.149409f,  0.195091f,  -0.033731f, 0.019861f,  -0.064047f, -0.471328f,
                -0.004093f, 0.016803f,  0.044635f,  0.058912f,  -0.018735f, 0.035536f,  -0.050373f, -0.002794f,
                -0.086705f, 0.038435f,  -0.301466f, -1.071246f, -0.028247f, 0.018984f,  0.254702f,  0.141142f,
                -0.017522f, 0.014843f,  0.079391f,  0.079662f,  -0.051204f, 0.048419f,  0.235604f,  -0.185797f,
                -0.019569f, 0.02678f,   0.162507f,  0.046435f,  -0.004606f, 0.08806f,   0.18634f,   0.193957f,
                -0.024333f, -0.01298f,  -0.17977f,  -0.65881f,  -0.003778f, 0.007418f,  0.065439f,  0.104549f,
                -0.027706f, 0.03301f,   0.057492f,  0.032019f,  -0.135337f, 0.000269f,  -0.250203f, -1.181688f,
                -0.027022f, -0.006755f, 0.206848f,  0.129268f,  -0.003529f, 0.013445f,  0.181484f,  0.139955f,
                -0.036587f, 0.065824f,  0.288751f,  -0.110813f, -0.015578f, 0.044818f,  0.17756f,   0.006914f,
                0.002329f,  0.068982f,  0.189079f,  0.184253f,  0.00301f,   -0.039168f, -0.010855f, -0.393254f,
                0.000028f,  0.001906f,  0.07217f,   0.063305f,  -0.026144f, 0.028842f,  0.139149f,  0.023377f,
                0.023362f,  0.023559f,  -0.145386f, -0.863572f, -0.015749f, -0.021364f, 0.172571f,  0.078393f,
                -0.037253f, 0.014978f,  0.221502f,  0.189111f,  -0.048956f, 0.085409f,  0.325399f,  -0.058294f,
                -0.028495f, 0.021663f,  0.19392f,   0.02706f,   0.006908f,  0.065751f,  0.176395f,  0.138375f,
                0.012418f,  -0.031228f, -0.008762f, -0.427345f, -0.013677f, -0.002429f, 0.069655f,  0.019505f,
                -0.036763f, 0.022528f,  0.201062f,  0.022205f,  0.024528f,  0.06241f,   -0.076237f, -0.840695f,
                -0.007268f, -0.027865f, 0.211056f,  0.074744f,  -0.053563f, 0.006863f,  0.301432f,  0.192879f,
                -0.021944f, 0.100535f,  0.19031f,   -0.133746f, -0.006151f, 0.023944f,  0.13561f,   -0.03259f,
                0.000618f,  0.063736f,  0.180904f,  0.12393f,   0.001275f,  -0.0306f,   -0.032822f, -0.496515f,
                0.009757f,  0.014602f,  0.004532f,  -0.039969f, -0.015984f, 0.047726f,  0.099865f,  0.003163f,
                0.026623f,  0.117951f,  -0.076234f, -0.811997f, 0.01301f,   0.020042f,  0.173756f,  -0.036191f,
                -0.068887f, 0.0229f,    0.245465f,  0.214282f,  -0.011054f, 0.132813f,  0.241014f,  -0.148763f,
            },
            std::vector<T>{210, 350, 1},
            std::vector<T>{
                0.000000f, 0.000000f,   0.000000f,  349.000000f, 209.000000f,  // 0
                0.000000f, 0.000000f,   0.000000f,  237.625443f, 209.000000f,  // 5
                0.000000f, 140.305511f, 0.000000f,  349.000000f, 209.000000f,  // 10
                0.000000f, 0.000000f,   0.000000f,  349.000000f, 65.359818f,   // 15
                0.000000f, 0.000000f,   0.000000f,  349.000000f, 130.324097f,  // 20
                0.000000f, 0.000000f,   15.562508f, 97.587891f,  181.224182f,  // 25
                0.000000f, 0.000000f,   68.539543f, 250.406708f, 209.000000f,  // 30
                0.000000f, 0.000000f,   0.000000f,  195.881531f, 99.841385f,   // 35
                0.000000f, 0.000000f,   0.000000f,  78.303986f,  209.000000f,  // 40
                0.000000f, 0.000000f,   0.000000f,  0.000000f,   209.000000f,  // 45
            },
            std::vector<T>{
                0.3091790f,
                0.1555300f,
                0.1549760f,
                0.1466370f,
                0.0260620f,
                0.0177700f,
                0.0019870f,
                0.0008570f,
                0.0002190f,
                0.0000000f,
            },
            ""},
        ProposalV4Params{
            0.7f,     // iou_threshold
            16,       // min_bbox_size
            16,       // feature_stride
            6000,     // pre_nms_topn
            10,       // post_nms_topn
            {0.5f},   // ratios
            {32.0f},  // scales
            1,        // batch_size
            1,        // anchor_num
            10,       // feat_map_height
            10,       // feat_map_width
            IN_ET,
            std::vector<T>{
                0.000240f, 0.003802f, 0.111432f, 0.000503f, 0.007887f, 0.144701f, 0.399074f, 0.004680f,  // 0
                0.139741f, 0.002386f, 0.030003f, 0.276552f, 0.000267f, 0.022971f, 0.287953f, 0.050235f,  // 8
                0.002580f, 0.206311f, 0.000146f, 0.009656f, 0.175462f, 0.000147f, 0.014718f, 0.272348f,  // 16
                0.065199f, 0.003286f, 0.185335f, 0.003720f, 0.025932f, 0.251401f, 0.001465f, 0.090447f,  // 24
                0.488469f, 0.092259f, 0.019306f, 0.379091f, 0.005311f, 0.010369f, 0.087615f, 0.042003f,  // 32
                0.073871f, 0.416763f, 0.044282f, 0.069776f, 0.313032f, 0.000457f, 0.017346f, 0.089762f,  // 40
                0.000820f, 0.103986f, 0.367993f, 0.026315f, 0.035701f, 0.299252f, 0.000135f, 0.017825f,  // 48
                0.150119f, 0.000076f, 0.050511f, 0.269601f, 0.026680f, 0.003541f, 0.189765f, 0.000051f,  // 56
                0.004315f, 0.193150f, 0.000032f, 0.007254f, 0.185557f, 0.051526f, 0.000657f, 0.117579f,  // 64
                0.000115f, 0.010179f, 0.293187f, 0.000025f, 0.006505f, 0.175345f, 0.032587f, 0.000469f,  // 72
                0.098443f, 0.000121f, 0.009600f, 0.322782f, 0.000032f, 0.004543f, 0.166860f, 0.044911f,  // 80
                0.000187f, 0.102691f, 0.000242f, 0.005502f, 0.107865f, 0.000191f, 0.005336f, 0.086893f,  // 88
                0.078422f, 0.000345f, 0.079096f, 0.000281f, 0.016388f, 0.214072f, 0.000107f, 0.012027f,  // 96
                0.192754f, 0.049531f, 0.000386f, 0.149893f, 0.000374f, 0.016965f, 0.204781f, 0.000163f,  // 104
                0.016272f, 0.215277f, 0.032298f, 0.000857f, 0.133426f, 0.000614f, 0.020215f, 0.165789f,  // 112
                0.000225f, 0.036951f, 0.262195f, 0.087675f, 0.004596f, 0.147764f, 0.000219f, 0.010502f,  // 120
                0.163394f, 0.000152f, 0.023116f, 0.241702f, 0.081800f, 0.002197f, 0.146637f, 0.000193f,  // 128
                0.012017f, 0.133497f, 0.000375f, 0.028605f, 0.309179f, 0.065962f, 0.005508f, 0.155530f,  // 136
                0.000186f, 0.004540f, 0.079319f, 0.000799f, 0.031003f, 0.303045f, 0.051473f, 0.017770f,  // 144
                0.206188f, 0.000202f, 0.004291f, 0.061095f, 0.001109f, 0.018094f, 0.156639f, 0.026062f,  // 152
                0.005270f, 0.148651f, 0.000026f, 0.007300f, 0.096013f, 0.000383f, 0.022134f, 0.129511f,  // 160
                0.080882f, 0.003416f, 0.129922f, 0.000037f, 0.010040f, 0.130007f, 0.000116f, 0.014904f,  // 168
                0.171423f, 0.082893f, 0.000921f, 0.154976f, 0.000142f, 0.016552f, 0.209696f, 0.000227f,  // 176
                0.022418f, 0.228501f, 0.111712f, 0.001987f, 0.158164f, 0.001200f, 0.027049f, 0.308222f,  // 184
                0.001366f, 0.038146f, 0.287945f, 0.072526f, 0.016064f, 0.257895f, 0.000595f, 0.016962f,  // 192
            },
            std::vector<T>{
                0.006756f,  -0.055635f, 0.030843f,  0.007482f,  0.009056f,  -0.041824f, 0.119722f,  0.168988f,
                0.002822f,  0.039733f,  0.109005f,  0.245152f,  -0.013196f, -0.018222f, -0.170122f, -0.374904f,
                -0.005455f, -0.034059f, -0.006787f, 0.072005f,  -0.017933f, -0.007358f, 0.034149f,  0.123846f,
                0.128319f,  0.016107f,  -0.615487f, -1.235094f, -0.024253f, -0.019406f, 0.134142f,  0.157853f,
                -0.021119f, 0.007383f,  0.089365f,  0.092854f,  0.062491f,  0.002366f,  0.122464f,  -0.003326f,
                0.015468f,  -0.034088f, 0.079009f,  0.075483f,  0.011972f,  0.042427f,  0.106865f,  0.158754f,
                0.071211f,  -0.034009f, 0.007985f,  -0.441477f, 0.009046f,  -0.028515f, 0.095372f,  0.119598f,
                -0.007553f, -0.0072f,   0.105072f,  0.084314f,  0.23268f,   -0.02906f,  -0.408454f, -1.13439f,
                0.016202f,  -0.037859f, 0.130873f,  0.129652f,  0.002064f,  -0.011969f, 0.171623f,  0.050218f,
                0.113831f,  0.028922f,  0.017785f,  0.059708f,  0.037658f,  -0.011245f, 0.097197f,  0.137491f,
                0.024218f,  0.04739f,   0.091978f,  0.217333f,  0.088418f,  -0.004662f, -0.095168f, -0.397928f,
                0.02639f,   -0.008501f, 0.068487f,  0.108465f,  0.020069f,  0.018829f,  0.040206f,  0.068473f,
                0.226458f,  -0.072871f, -0.672384f, -1.447558f, 0.039598f,  0.017471f,  0.187288f,  0.08409f,
                0.017152f,  -0.00516f,  0.183419f,  0.068469f,  0.063944f,  0.160725f,  -0.022493f, -0.132291f,
                0.010542f,  0.036318f,  0.074042f,  -0.013323f, 0.00808f,   0.060365f,  0.120566f,  0.21866f,
                0.046324f,  0.088741f,  0.029469f,  -0.517183f, 0.00917f,   0.011915f,  0.053674f,  0.140168f,
                0.0033f,    0.022759f,  -0.006196f, 0.063839f,  0.083726f,  -0.088385f, -0.57208f,  -1.454211f,
                0.020655f,  0.010788f,  0.134951f,  0.109709f,  0.015445f,  -0.015363f, 0.109153f,  0.051209f,
                0.024297f,  0.139126f,  -0.12358f,  -0.127979f, 0.004587f,  0.004751f,  0.047292f,  0.027066f,
                0.011003f,  0.069887f,  0.117052f,  0.267419f,  0.039306f,  0.077584f,  0.02579f,   -0.496149f,
                -0.005569f, 0.015494f,  -0.011662f, 0.105549f,  -0.007015f, 0.031984f,  -0.075742f, 0.0852f,
                0.023886f,  -0.053107f, -0.325533f, -1.329066f, 0.004688f,  0.034501f,  0.089317f,  0.042463f,
                0.004212f,  -0.015128f, 0.00892f,   0.028266f,  0.009997f,  0.157822f,  0.020116f,  -0.142337f,
                0.008199f,  0.046564f,  0.083014f,  0.046307f,  0.006771f,  0.084997f,  0.141935f,  0.228339f,
                -0.020308f, 0.077745f,  -0.018319f, -0.522311f, 0.010432f,  0.024641f,  0.020571f,  0.097148f,
                0.002064f,  0.035053f,  -0.121995f, 0.012222f,  -0.030779f, 0.100481f,  -0.331737f, -1.257669f,
                -0.013079f, 0.021227f,  0.159949f,  0.120097f,  0.005765f,  -0.012335f, -0.005268f, 0.042067f,
                -0.043972f, 0.102556f,  0.180494f,  -0.084721f, -0.011962f, 0.031302f,  0.112511f,  0.027557f,
                -0.002085f, 0.082978f,  0.149409f,  0.195091f,  -0.033731f, 0.019861f,  -0.064047f, -0.471328f,
                -0.004093f, 0.016803f,  0.044635f,  0.058912f,  -0.018735f, 0.035536f,  -0.050373f, -0.002794f,
                -0.086705f, 0.038435f,  -0.301466f, -1.071246f, -0.028247f, 0.018984f,  0.254702f,  0.141142f,
                -0.017522f, 0.014843f,  0.079391f,  0.079662f,  -0.051204f, 0.048419f,  0.235604f,  -0.185797f,
                -0.019569f, 0.02678f,   0.162507f,  0.046435f,  -0.004606f, 0.08806f,   0.18634f,   0.193957f,
                -0.024333f, -0.01298f,  -0.17977f,  -0.65881f,  -0.003778f, 0.007418f,  0.065439f,  0.104549f,
                -0.027706f, 0.03301f,   0.057492f,  0.032019f,  -0.135337f, 0.000269f,  -0.250203f, -1.181688f,
                -0.027022f, -0.006755f, 0.206848f,  0.129268f,  -0.003529f, 0.013445f,  0.181484f,  0.139955f,
                -0.036587f, 0.065824f,  0.288751f,  -0.110813f, -0.015578f, 0.044818f,  0.17756f,   0.006914f,
                0.002329f,  0.068982f,  0.189079f,  0.184253f,  0.00301f,   -0.039168f, -0.010855f, -0.393254f,
                0.000028f,  0.001906f,  0.07217f,   0.063305f,  -0.026144f, 0.028842f,  0.139149f,  0.023377f,
                0.023362f,  0.023559f,  -0.145386f, -0.863572f, -0.015749f, -0.021364f, 0.172571f,  0.078393f,
                -0.037253f, 0.014978f,  0.221502f,  0.189111f,  -0.048956f, 0.085409f,  0.325399f,  -0.058294f,
                -0.028495f, 0.021663f,  0.19392f,   0.02706f,   0.006908f,  0.065751f,  0.176395f,  0.138375f,
                0.012418f,  -0.031228f, -0.008762f, -0.427345f, -0.013677f, -0.002429f, 0.069655f,  0.019505f,
                -0.036763f, 0.022528f,  0.201062f,  0.022205f,  0.024528f,  0.06241f,   -0.076237f, -0.840695f,
                -0.007268f, -0.027865f, 0.211056f,  0.074744f,  -0.053563f, 0.006863f,  0.301432f,  0.192879f,
                -0.021944f, 0.100535f,  0.19031f,   -0.133746f, -0.006151f, 0.023944f,  0.13561f,   -0.03259f,
                0.000618f,  0.063736f,  0.180904f,  0.12393f,   0.001275f,  -0.0306f,   -0.032822f, -0.496515f,
                0.009757f,  0.014602f,  0.004532f,  -0.039969f, -0.015984f, 0.047726f,  0.099865f,  0.003163f,
                0.026623f,  0.117951f,  -0.076234f, -0.811997f, 0.01301f,   0.020042f,  0.173756f,  -0.036191f,
                -0.068887f, 0.0229f,    0.245465f,  0.214282f,  -0.011054f, 0.132813f,  0.241014f,  -0.148763f,
            },
            std::vector<T>{210, 350, 1, 1},
            std::vector<T>{0.f,      11.9688f, 4.02532f, 204.528f, 182.586f, 0.f,      33.7915f, 48.4886f, 210.f,
                           238.505f, 0.f,      0.f,      0.f,      204.428f, 337.029f, 0.f,      72.611f,  9.87545f,
                           203.687f, 212.299f, 0.f,      5.08432f, 4.19913f, 208.719f, 249.225f, 0.f,      23.6503f,
                           57.8165f, 210.f,    350.f,    0.f,      84.8804f, 9.47241f, 156.822f, 243.003f, 0.f,
                           101.663f, 15.5542f, 166.083f, 327.839f, 0.f,      13.9738f, 0.f,      210.f,    128.482f,
                           0.f,      77.8929f, 29.663f,  186.561f, 313.287f

            },
            std::vector<
                T>{0.309179, 0.308222, 0.303045, 0.241702, 0.192754, 0.165789, 0.15553, 0.154976, 0.146637, 0.129511},
            "tensorflow"},
    };
    return proposalV4Params;
}

std::vector<ProposalV1Params> generateProposalV1CombinedParams() {
    std::vector<std::vector<ProposalV1Params>> proposalTypeParams{generateProposalV1Params<element::Type_t::f64>(),
                                                                  generateProposalV1Params<element::Type_t::f32>(),
                                                                  generateProposalV1Params<element::Type_t::f16>(),
                                                                  generateProposalV1Params<element::Type_t::bf16>()};
    std::vector<ProposalV1Params> combinedParams;
    for (auto& params : proposalTypeParams)
        std::move(params.begin(), params.end(), std::back_inserter(combinedParams));
    return combinedParams;
}

std::vector<ProposalV4Params> generateProposalV4CombinedParams() {
    std::vector<std::vector<ProposalV4Params>> proposalTypeParams{generateProposalV4Params<element::Type_t::f64>(),
                                                                  generateProposalV4Params<element::Type_t::f32>(),
                                                                  generateProposalV4Params<element::Type_t::f16>(),
                                                                  generateProposalV4Params<element::Type_t::bf16>()};
    std::vector<ProposalV4Params> combinedParams;
    for (auto& params : proposalTypeParams)
        std::move(params.begin(), params.end(), std::back_inserter(combinedParams));
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Proposal_With_Hardcoded_Refs,
                         ReferenceProposalV1LayerTest,
                         testing::ValuesIn(generateProposalV1CombinedParams()),
                         ReferenceProposalV1LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Proposal_With_Hardcoded_Refs,
                         ReferenceProposalV4LayerTest,
                         testing::ValuesIn(generateProposalV4CombinedParams()),
                         ReferenceProposalV4LayerTest::getTestCaseName);

}  // namespace
