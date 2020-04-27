// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <gtest/gtest.h>
#include "myriad_layers_tests.hpp"
#include <blob_factory.hpp>

using namespace InferenceEngine;

#define PROPOSAL_ELEMENT_SIZE (5)   // batch_num, left, top, right, bottom
#define OUTPUT_SAMPLING_NUM   (20)  // Validate only top 20 rois
#define OUTPUT_ROI_MATCH_THRESHOLD   (18)  // At least 18 rois should be matched

class myriadLayersTestsProposal_nightly : public myriadLayersTests_nightly {

protected:
 std::string model;
 std::string cls_prob_file;
 std::string bbox_pred_file;
 Blob::Ptr outputBlob;

 //TODO: make test_p
 std::string clip_before_nms="true";
 std::string clip_after_nms="false";
 std::string normalize="false";


 /**
  * to generate reference mkldnn-can be used, but even in hetero cpu,cpu it cannot handle fp16 input
  */
 Precision precision = Precision::FP16; // (or FP32)

 public:
    void InferProposalLayer() {
        SetSeed(DEFAULT_SEED_VALUE + 13);

        StatusCode st;

        REPLACE_WITH_STR(model, "__PRECISION__", precision.name());

        REPLACE_WITH_STR(model, "__CLIP_BEFORE_NMS__", clip_before_nms);
        REPLACE_WITH_STR(model, "__CLIP_AFTER_NMS__", clip_after_nms);
        REPLACE_WITH_STR(model, "__NORMALIZE__", normalize);

        ASSERT_NO_THROW(readNetwork(model));

        const auto& network = _cnnNetwork;

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["rpn_cls_prob_reshape"]->setPrecision(precision);
        _inputsInfo["rpn_cls_prob_reshape"]->setLayout(NCHW);
        _inputsInfo["rpn_bbox_pred"]->setPrecision(precision);
        _inputsInfo["rpn_bbox_pred"]->setLayout(NCHW);
        _inputsInfo["im_info"]->setPrecision(precision);

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo["proposal"]->setPrecision(precision);
        _outputsInfo["proposal"]->setLayout(NC);

        ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network, {}, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

        ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr rpn_cls_prob_reshape;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("rpn_cls_prob_reshape", rpn_cls_prob_reshape, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr rpn_bbox_pred;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("rpn_bbox_pred", rpn_bbox_pred, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr img_info;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("im_info", img_info, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        // for rpn_cls_prob_reshape
        std::string inputTensor1Binary = TestDataHelpers::get_data_path() + cls_prob_file;
        ASSERT_TRUE(fromBinaryFile(inputTensor1Binary, rpn_cls_prob_reshape));

        // rpn_bbox_pred
        std::string inputTensor2Binary = TestDataHelpers::get_data_path() + bbox_pred_file;
        ASSERT_TRUE(fromBinaryFile(inputTensor2Binary, rpn_bbox_pred));

        if (precision == Precision::FP16) {
            ie_fp16 *img_info_data = img_info->buffer().as<ie_fp16 *>();
            img_info_data[0] = PrecisionUtils::f32tof16(224.f);
            img_info_data[1] = PrecisionUtils::f32tof16(224.f);
            img_info_data[2] = PrecisionUtils::f32tof16(1.0f);
        }
        if (precision == Precision::FP32) {
            float *img_info_data = img_info->buffer().as<float *>();
            img_info_data[0] = 224.f;
            img_info_data[1] = 224.f;
            img_info_data[2] = 1.0f;
        }


        ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(_inferRequest->GetBlob("proposal", outputBlob, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    }

    void compareOutputSampleToRef(std::vector<float> & gt_values, const float error_threshold) {
        const int gt_values_size = gt_values.size();
        auto tmpTensorDesc = outputBlob->getTensorDesc();
        tmpTensorDesc.getDims()[tmpTensorDesc.getDims().size() - 2] = OUTPUT_SAMPLING_NUM;

        // Generate GT
        _refBlob = make_blob_with_precision(tmpTensorDesc);
        _refBlob->allocate();

        if (precision != Precision::FP32) {
            std::transform(std::begin(gt_values), std::end(gt_values), _refBlob->buffer().as<ie_fp16 *>(),
                           [](float gt) -> ie_fp16 { return PrecisionUtils::f32tof16(gt); });
        } else {
            std::copy(std::begin(gt_values), std::end(gt_values), _refBlob->buffer().as<float *>());
        }

        // Sampling top 20 results from output
        Blob::Ptr outputSample = make_blob_with_precision(tmpTensorDesc);
        outputSample->allocate();

        if (precision != Precision::FP32)
            std::copy_n(outputBlob->cbuffer().as<const ie_fp16*>(), gt_values_size, outputSample->buffer().as<ie_fp16*>());
        else
            std::copy_n(outputBlob->cbuffer().as<const float *>(), gt_values_size, outputSample->buffer().as<float *>());

        CompareCommonAbsolute(outputSample, _refBlob, error_threshold);
    }

    int calcIoU(std::vector<float> &gt_values) {
        // Validate top 20 rois with GT
        const float iou_threshold = 0.7f;
        const float num_gt_rois = 100;
        const int num_roi_elem = 4;  // ignore score value, only get rois
        const auto actual_ptr = outputBlob->cbuffer().as<ie_fp16*>();
        const auto actual_ptr_float = outputBlob->cbuffer().as<float*>();
        int matched_count = 0;
        for (int i = 0; i < OUTPUT_SAMPLING_NUM; i++)  // Verify only Top 20 rois with highest scores
        {
            float actual_values[num_roi_elem];
            for (int j = 0; j < num_roi_elem; j++)
            {
                if (precision == Precision::FP16)
                    actual_values[j] = PrecisionUtils::f16tof32(actual_ptr[i*PROPOSAL_ELEMENT_SIZE + (j+1)]);
                else
                    actual_values[j] = actual_ptr_float[i*PROPOSAL_ELEMENT_SIZE + (j+1)];
            }

            float max_iou = 0.f;
            for (int j = 0; j < num_gt_rois; j++)
            {
                float cur_iou = check_iou(&actual_values[0], &gt_values[j*PROPOSAL_ELEMENT_SIZE+1]);  // start index 1 to ignore score value
                if (cur_iou > max_iou)
                {
                    max_iou = cur_iou;
                }
            }

            if (max_iou > iou_threshold)
            {
                matched_count++;
            }
        }

        return matched_count;
    }
    /**
    * "IoU = intersection area / union area" of two boxes A, B
    *   A, B: 4-dim array (x1, y1, x2, y2)
    */
    static float check_iou(const float* A, const float* B)
    {
        if (A[0] > B[2] || A[1] > B[3] || A[2] < B[0] || A[3] < B[1]) {
            return 0;
        }
        else {
            // overlapped region (= box)
            const float x1 = std::max(A[0], B[0]);
            const float y1 = std::max(A[1], B[1]);
            const float x2 = std::min(A[2], B[2]);
            const float y2 = std::min(A[3], B[3]);

            // intersection area
            const float width = std::max(0.0f, x2 - x1 + 1.0f);
            const float height = std::max(0.0f, y2 - y1 + 1.0f);
            const float area = width * height;

            // area of A, B
            const float A_area = (A[2] - A[0] + 1.0f) * (A[3] - A[1] + 1.0f);
            const float B_area = (B[2] - B[0] + 1.0f) * (B[3] - B[1] + 1.0f);

            // IoU
            return area / (A_area + B_area - area);
        }
    }

};

std::string caffeModel() {
    return R"V0G0N(
        <net name="testProposal" version="2" batch="1">
            <layers>
                <layer id="0" name="rpn_cls_prob_reshape" precision="__PRECISION__" type="Input">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>18</dim>
                            <dim>14</dim>
                            <dim>14</dim>
                        </port>
                    </output>
                </layer>
                <layer id="1" name="rpn_bbox_pred" precision="__PRECISION__" type="Input">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>36</dim>
                            <dim>14</dim>
                            <dim>14</dim>
                        </port>
                    </output>
                </layer>
                <layer id="2" name="im_info" precision="__PRECISION__" type="Input">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>3</dim>
                        </port>
                    </output>
                </layer>
                <layer id="3" name="proposal" precision="__PRECISION__" type="Proposal">
                    <data base_size="16" feat_stride="16" min_size="16" nms_thresh="0.7" post_nms_topn="300"
                     pre_nms_topn="6000" ratio="0.5,1,2" scale="8,16,32" pre_nms_thresh="0.0"
                     clip_before_nms="__CLIP_BEFORE_NMS__"
                     clip_after_nms="__CLIP_AFTER_NMS__"
                     normalize="__NORMALIZE__" />
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>18</dim>
                            <dim>14</dim>
                            <dim>14</dim>
                        </port>
                        <port id="1">
                            <dim>1</dim>
                            <dim>36</dim>
                            <dim>14</dim>
                            <dim>14</dim>
                        </port>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>300</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
                <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
                <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
            </edges>
        </net>
    )V0G0N";
}

TEST_F(myriadLayersTestsProposal_nightly, Caffe) {

    // Verify only 20 ranked proposal output with GT values
    std::vector<float> gt_values = {
        0.f,     72.363f,  58.942f,  197.141f, 177.96f,  // batch_num, left, top, right, bottom
        0.f,      6.885f,  75.163f,  50.71f,   144.593f,
        0.f,     14.197f,  94.857f,  35.119f,  149.677f,
        0.f,     22.245f,  66.52f,   223.f,    170.073f,
        0.f,      9.534f,  68.92f,   45.507f,  135.162f,
        0.f,     20.249f,  83.337f,  43.703f,  141.187f,
        0.f,     56.778f,  48.152f,  202.341f, 204.147f,
        0.f,      0.f,     34.429f,  223.f,    195.942f,
        0.f,     13.293f,  76.712f,  69.431f,  171.435f,
        0.f,     12.007f,  71.093f,  54.662f,  162.261f,
        0.f,     68.36f,   90.737f,  210.758f, 175.72f ,
        0.f,    107.295f,  114.45f,  174.194f, 155.064f,
        0.f,     22.078f,  84.345f,  136.421f, 217.647f,
        0.f,     80.583f,  76.407f,  187.699f, 146.992f,
        0.f,     35.923f,  78.907f,  209.749f, 159.349f,
        0.f,      0.f,     69.083f,  223.f,    223.f   ,
        // NOTE : on CPU this two proposal boxes have been swapped
        0.f,     10.249f,  55.264f,  142.18f,  192.755f,
        0.f,    109.44f,   120.758f, 177.096f, 161.314f,

        0.f,     47.69f,   81.621f,  179.488f, 152.899f,
        0.f,     24.46f,   52.069f,  73.301f,  151.432f};


    cls_prob_file  = "/vpu/proposal_input_rpn_cls_prob_reshape.bin";
    bbox_pred_file = "/vpu/proposal_input_rpn_bbox_pred.bin";
    model          = caffeModel();

    ASSERT_NO_FATAL_FAILURE(InferProposalLayer());
    ASSERT_NO_FATAL_FAILURE(compareOutputSampleToRef(gt_values, 0.26f));
}

TEST_F(myriadLayersTestsProposal_nightly, CaffeNoClipBeforeNms) {

    // Verify only 20 ranked proposal output with GT values - reference get from MKLDNN plugin
    std::vector<float> gt_values = {
        0, 72.408f,   58.925f,  197.062f, 177.856f,
        0, 6.907f,    75.193f,  50.726f,  144.589f,
        0, 14.264f,   94.921f,  35.206f,  149.710f,
        0, 22.460f,   66.425f,  250.893f, 170.004f,
        0, 9.483f,    68.801f,  45.493f,  135.213f,
        0, 20.271f,   83.327f,  43.680f,  141.196f,
        0, 13.909f,   51.622f,  223.455f, 191.746f,
        0, 56.799f,   48.177f,  202.286f, 203.900f,
        0, -72.978f,  34.416f,  343.053f, 195.997f,
        0, -107.360f, 52.610f,  326.355f, 222.650f,
        0, -223.833f, 21.986f,  428.385f, 299.044f,

        // swapped on mkldnn
        0, 13.290f,   76.607f,  69.404f,  171.320f,
        0, 12.085f,   71.267f,  54.660f,  162.396f,

        0, -361.772f, 56.140f,  531.011f, 291.391f,
        0, 68.313f,   90.692f,  210.803f, 175.800f,
        0, 107.404f,  114.274f, 174.122f, 154.990f,
        0, 22.029f,   84.392f,  136.439f, 217.658f,
        0, 80.502f,   76.457f,  187.633f, 147.042f,
        0, -253.904f, 4.927f,   451.118f, 235.572f,
        0, 36.144f,   78.806f,  209.537f, 159.228f};


    // Test settings
    cls_prob_file   = "/vpu/proposal_input_rpn_cls_prob_reshape.bin";
    bbox_pred_file  = "/vpu/proposal_input_rpn_bbox_pred.bin";
    clip_before_nms = "false";
    model           = caffeModel();

    ASSERT_NO_FATAL_FAILURE(InferProposalLayer());
    ASSERT_NO_FATAL_FAILURE(compareOutputSampleToRef(gt_values, 0.26f));
}

TEST_F(myriadLayersTestsProposal_nightly, CaffeClipAfterNms) {

    // Verify only 20 ranked proposal output with GT values
    std::vector<float> gt_values = {
        0, 72.408f,  58.925f, 197.062f, 177.856f,
        0, 6.907f,   75.193f, 50.726f, 144.589f,
        0, 14.264f,  94.921f, 35.206f, 149.710f,
        0, 22.460f,  66.425f, 223.f, 170.004f,
        0, 9.483f,   68.801f, 45.493f, 135.213f,
        0, 20.271f,  83.327f, 43.680f, 141.196f,
        0, 56.799f,  48.177f, 202.286f, 203.900f,
        0, 0,        34.416f, 223.f, 195.997f,

        //swapped on CPU
        0, 13.290f,  76.607f, 69.404f, 171.320f,
        0, 12.085f,  71.267f, 54.660f, 162.396f,

        0, 68.313f,  90.692f, 210.803f, 175.800f,
        0, 107.404f, 114.274f, 174.122f, 154.990f,
        0, 22.029f,  84.392f, 136.439f, 217.658f,
        0, 80.502f,  76.457f, 187.633f, 147.042f,
        0, 36.144f,  78.806f, 209.537f, 159.228f,
        0, 0,        68.821f, 223.f, 223.f,

        //swapped on CPU
        0, 10.207f,  55.363f, 142.246f, 192.734f,
        0, 109.509f, 120.650f, 176.948f, 161.247f,

        0, 47.612f,  81.498f, 179.434f, 152.845f,
        0, 24.402f,  51.990f, 73.238f, 151.443f};


    // Test settings
    cls_prob_file   = "/vpu/proposal_input_rpn_cls_prob_reshape.bin";
    bbox_pred_file  = "/vpu/proposal_input_rpn_bbox_pred.bin";
    clip_after_nms  = "true";
    model           = caffeModel();

    ASSERT_NO_FATAL_FAILURE(InferProposalLayer());
    ASSERT_NO_FATAL_FAILURE(compareOutputSampleToRef(gt_values, 0.26f));
}

TEST_F(myriadLayersTestsProposal_nightly, CaffeNormalizedOutput) {

    // Verify only 20 ranked proposal output with GT values
    std::vector<float> gt_values = {
        0, 0.323f, 0.263f, 0.879f, 0.794f,
        0, 0.030f, 0.335f, 0.226f, 0.645f,
        0, 0.063f, 0.423f, 0.157f, 0.668f,
        0, 0.100f, 0.296f, 0.995f, 0.758f,
        0, 0.042f, 0.307f, 0.203f, 0.603f,
        0, 0.090f, 0.371f, 0.195f, 0.630f,
        0, 0.253f, 0.215f, 0.903f, 0.910f,
        0, 0,      0.153f, 0.995f, 0.874f,

        // swapped on CPU
        0, 0.059f, 0.341f, 0.309f, 0.764f,
        0, 0.053f, 0.318f, 0.244f, 0.724f,

        0, 0.304f, 0.404f, 0.941f, 0.784f,
        0, 0.479f, 0.510f, 0.777f, 0.691f,
        0, 0.098f, 0.376f, 0.609f, 0.971f,
        0, 0.359f, 0.341f, 0.837f, 0.656f,
        0, 0.161f, 0.351f, 0.935f, 0.710f,

        //swapped on CPU
        0, 0,      0.307f, 0.995f, 0.995f,
        0, 0.045f, 0.247f, 0.635f, 0.860f,

        0, 0.488f, 0.538f, 0.789f, 0.719f,
        0, 0.212f, 0.363f, 0.801f, 0.682f,
        0, 0.108f, 0.232f, 0.326f, 0.676f};


    // Test settings
    cls_prob_file   = "/vpu/proposal_input_rpn_cls_prob_reshape.bin";
    bbox_pred_file  = "/vpu/proposal_input_rpn_bbox_pred.bin";
    normalize       = "true";
    model           = caffeModel();

    ASSERT_NO_FATAL_FAILURE(InferProposalLayer());
    ASSERT_NO_FATAL_FAILURE(compareOutputSampleToRef(gt_values, 0.026f));
}

TEST_F(myriadLayersTestsProposal_nightly, TensorFlow) {

     model = R"V0G0N(
        <net name="testProposal" version="2" batch="1">
            <layers>
                <layer id="0" name="rpn_cls_prob_reshape" precision="__PRECISION__" type="Input">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>24</dim>
                            <dim>14</dim>
                            <dim>14</dim>
                        </port>
                    </output>
                </layer>
                <layer id="1" name="rpn_bbox_pred" precision="__PRECISION__" type="Input">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>48</dim>
                            <dim>14</dim>
                            <dim>14</dim>
                        </port>
                    </output>
                </layer>
                <layer id="2" name="im_info" precision="__PRECISION__" type="Input">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>3</dim>
                        </port>
                    </output>
                </layer>
                <layer id="3" name="proposal" precision="__PRECISION__" type="Proposal">
                    <data base_size="256" box_coordinate_scale="10" box_size_scale="5" feat_stride="16" framework="tensorflow" min_size="10" nms_thresh="0.7" post_nms_topn="100" pre_nms_topn="21474" ratio="0.5,1.0,2.0" scale="0.25,0.5,1.0,2.0"/>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>24</dim>
                            <dim>14</dim>
                            <dim>14</dim>
                        </port>
                        <port id="1">
                            <dim>1</dim>
                            <dim>48</dim>
                            <dim>14</dim>
                            <dim>14</dim>
                        </port>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>100</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
                <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
                <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
            </edges>
        </net>
    )V0G0N";

    // Verify only 20 ranked proposal output is existed in GT values
    std::vector<float> gt_values = {
        0.f, 53.8881f, 29.2742f, 195.151f, 176.956f,  // batch_num, left, top, right, bottom
        0.f, 56.9245f, 97.0252f, 191.764f, 211.743f,
        0.f, 53.7577f, 10.861f,  198.255f, 145.085f,
        0.f, 66.8727f, 19.5093f, 188.542f, 207.571f,
        0.f, 72.5254f, 17.1535f, 114.375f, 36.867f,
        0.f, 74.977f,  16.4198f, 115.675f, 31.661f,
        0.f, 75.7093f, 14.6141f, 129.408f, 30.7164f,
        0.f, 68.5911f, 2.17764f, 116.854f, 23.2988f,
        0.f, 44.069f,  119.669f, 187.943f, 207.186f,
        0.f, 73.7968f, 27.3531f, 120.728f, 43.4836f,
        0.f, 65.3355f, 23.6799f, 196.25f,  127.208f,
        0.f, 72.9499f, 24.7654f, 109.859f, 40.0457f,
        0.f, 73.9328f, 14.3182f, 129.78f,  37.398f,
        0.f, 71.6726f, 17.6392f, 103.571f, 32.9039f,
        0.f, 0.f,      0.677025f,224.f,    224.f,
        0.f, 80.1852f, 1.5176f,  112.255f, 13.5889f,
        0.f, 37.3883f, 11.9829f, 202.778f, 213.109f,
        0.f, 63.1822f, 129.085f, 96.5516f, 149.743f,
        0.f, 49.5382f, 137.022f, 178.434f, 214.141f,
        0.f, 90.8469f, 10.7634f, 206.273f, 125.817f,
        0.f, 52.0368f, 112.601f, 73.9616f, 130.725f,
        0.f, 60.5855f, 39.8242f, 208.905f, 221.897f,
        0.f, 46.8725f, 57.9966f, 203.359f, 162.118f,
        0.f, 42.5881f, 112.609f, 65.114f,  131.432f,
        0.f, 80.3221f, 37.2223f, 113.974f, 49.9118f,
        0.f, 93.5535f, 13.3071f, 153.779f, 43.266f,
        0.f, 70.2708f, 128.093f, 165.02f,  164.439f,
        0.f, 71.4465f, 20.636f,  86.9464f, 33.2511f,
        0.f, 67.2406f, 6.66637f, 98.914f,  28.4598f,
        0.f, 62.9305f, 19.6755f, 175.826f, 160.375f,
        0.f, 72.9378f, 17.9405f, 88.7149f, 28.9991f,
        0.f, 55.1827f, 149.366f, 174.829f, 206.638f,
        0.f, 54.9047f, 132.917f, 84.2464f, 147.54f,
        0.f, 67.0201f, 162.942f, 96.9201f, 196.988f,
        0.f, 102.893f, 15.4131f, 205.249f, 191.626f,
        0.f, 43.3537f, 40.8668f, 190.631f, 142.491f,
        0.f, 59.8731f, 137.814f, 169.612f, 200.622f,
        0.f, 86.5134f, 16.734f,  149.479f, 30.3793f,
        0.f, 65.7841f, 158.012f, 104.35f,  204.026f,
        0.f, 56.3348f, 130.639f, 101.328f, 150.503f,
        0.f, 110.973f, 11.6659f, 186.115f, 51.8234f,
        0.f, 56.8438f, 139.659f, 90.965f,  154.652f,
        0.f, 50.0912f, 10.1794f, 164.545f, 194.869f,
        0.f, 50.6326f, 116.361f, 68.4287f, 132.11f,
        0.f, 52.6678f, 11.2994f, 201.231f, 100.594f,
        0.f, 22.428f,  0.f,      180.848f, 194.578f,
        0.f, 121.799f, 10.7808f, 174.07f,  54.3841f,
        0.f, 61.9675f, 174.689f, 163.129f, 212.789f,
        0.f, 68.3703f, 86.9163f, 102.297f, 131.793f,
        0.f, 63.1688f, 141.842f, 144.259f, 209.627f,
        0.f, 91.0212f, 0.f,      143.93f,  22.4325f,
        0.f, 85.0419f, 42.8284f, 205.168f, 175.15f,
        0.f, 67.4256f, 131.442f, 98.0033f, 144.358f,
        0.f, 69.3273f, 106.276f, 96.8871f, 128.973f,
        0.f, 104.049f, 4.0801f,  124.051f, 20.6967f,
        0.f, 71.2328f, 69.3853f, 184.252f, 197.295f,
        0.f, 72.0265f, 0.426311f,96.3345f, 12.0004f,
        0.f, 55.4309f, 138.711f, 84.8696f, 152.147f,
        0.f, 63.7001f, 133.453f, 123.394f, 157.344f,
        0.f, 79.3643f, 34.2525f, 105.982f, 49.3115f,
        0.f, 59.3775f, 99.2007f, 104.55f,  139.959f,
        0.f, 126.697f, 5.20008f, 224.0f,   196.069f,
        0.f, 73.7945f, 86.5779f, 100.509f, 117.225f,
        0.f, 69.5673f, 168.975f, 100.984f, 208.11f,
        0.f, 67.1213f, 177.04f,  136.411f, 216.875f,
        0.f, 58.0965f, 147.441f, 81.145f,  158.661f,
        0.f, 59.6378f, 17.1767f, 109.574f, 50.6673f,
        0.f, 77.5341f, 19.9996f, 110.322f, 46.983f,
        0.f, 55.6317f, 49.5741f, 155.302f, 200.88f,
        0.f, 89.4005f, 126.625f, 147.32f,  147.27f,
        0.f, 81.515f,  127.61f,  103.404f, 148.08f,
        0.f, 47.3904f, 89.3484f, 204.269f, 179.917f,
        0.f, 84.4783f, 177.696f, 173.98f,  211.952f,
        0.f, 97.373f,  36.2838f, 115.779f, 53.6433f,
        0.f, 66.7269f, 126.805f, 108.757f, 144.899f,
        0.f, 53.6802f, 137.088f, 96.0449f, 152.658f,
        0.f, 39.4418f, 9.47529f, 188.078f, 83.2332f,
        0.f, 72.0867f, 25.4984f, 90.6464f, 39.8829f,
        0.f, 110.994f, 5.94071f, 162.54f,  32.1407f,
        0.f, 76.3537f, 183.958f, 155.425f, 215.032f,
        0.f, 79.7699f, 102.186f, 105.326f, 129.827f,
        0.f, 70.6935f, 36.0157f, 97.075f,  48.5611f,
        0.f, 47.4715f, 9.66474f, 201.332f, 49.1294f,
        0.f, 111.987f, 26.351f, 144.638f,  39.3359f,
        0.f, 59.4611f, 141.302f, 87.7403f, 152.782f,
        0.f, 70.3368f, 150.324f, 97.0087f, 178.19f,
        0.f, 82.5671f, 103.536f, 124.713f, 130.838f,
        0.f, 82.0366f, 189.618f, 110.221f, 205.794f,
        0.f, 64.6987f, 73.8598f, 107.236f, 131.602f,
        0.f, 100.596f, 192.156f, 125.981f, 207.468f,
        0.f, 44.5534f, 116.114f, 73.3897f, 137.017f,
        0.f, 84.5608f, 18.7706f, 140.066f, 39.775f,
        0.f, 83.5766f, 89.4399f, 103.842f, 112.501f,
        0.f, 49.7645f, 130.757f, 66.1343f, 145.153f,
        0.f, 48.9145f, 115.85f,  76.7334f, 134.724f,
        0.f, 68.791f,  2.68682f, 90.5532f, 20.4226f,
        0.f, 67.3277f, 175.467f, 108.399f, 215.786f,
        0.f, 53.4548f, 120.664f, 70.7046f, 139.624f,
        0.f, 49.5098f, 111.469f, 90.2394f, 136.238f,
        0.f, 75.1919f, 0.f,      223.661f, 211.529f};

    cls_prob_file = "/vpu/proposal_tf_input_Reshape_Permute_Class.bin";
    bbox_pred_file = "/vpu/proposal_tf_input_FirstStageBoxPredictor_BoxEncodingPredictor_Conv2D.bin";

    ASSERT_NO_FATAL_FAILURE(InferProposalLayer());
    ASSERT_GE(calcIoU(gt_values), OUTPUT_ROI_MATCH_THRESHOLD); // Passed if all top 20 rois are in gt rois.
}
