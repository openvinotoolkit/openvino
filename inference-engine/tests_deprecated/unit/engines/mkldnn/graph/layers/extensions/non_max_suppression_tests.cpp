// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"
#include <ie_core.hpp>


using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct nmsTF_test_params {
    int center_point_box;
    int sort_result_descending;
    InferenceEngine::SizeVector scoresDim;
    std::vector<float> boxes;
    std::vector<float> scores;
    std::vector<int> max_output_boxes_per_class;
    std::vector<float> iou_threshold;
    std::vector<float> score_threshold;

    int num_selected_indices;
    std::vector<int> ref;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

static float intersectionOverUnion(float* boxesI, float* boxesJ, bool center_point_box) {
    float yminI, xminI, ymaxI, xmaxI, yminJ, xminJ, ymaxJ, xmaxJ;
    if (center_point_box) {
        //  box format: x_center, y_center, width, height
        yminI = boxesI[1] - boxesI[3] / 2.f;
        xminI = boxesI[0] - boxesI[2] / 2.f;
        ymaxI = boxesI[1] + boxesI[3] / 2.f;
        xmaxI = boxesI[0] + boxesI[2] / 2.f;
        yminJ = boxesJ[1] - boxesJ[3] / 2.f;
        xminJ = boxesJ[0] - boxesJ[2] / 2.f;
        ymaxJ = boxesJ[1] + boxesJ[3] / 2.f;
        xmaxJ = boxesJ[0] + boxesJ[2] / 2.f;
    } else {
        //  box format: y1, x1, y2, x2
        yminI = (std::min)(boxesI[0], boxesI[2]);
        xminI = (std::min)(boxesI[1], boxesI[3]);
        ymaxI = (std::max)(boxesI[0], boxesI[2]);
        xmaxI = (std::max)(boxesI[1], boxesI[3]);
        yminJ = (std::min)(boxesJ[0], boxesJ[2]);
        xminJ = (std::min)(boxesJ[1], boxesJ[3]);
        ymaxJ = (std::max)(boxesJ[0], boxesJ[2]);
        xmaxJ = (std::max)(boxesJ[1], boxesJ[3]);
    }

    float areaI = (ymaxI - yminI) * (xmaxI - xminI);
    float areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
    if (areaI <= 0.f || areaJ <= 0.f)
        return 0.f;

    float intersection_area =
            (std::max)((std::min)(ymaxI, ymaxJ) - (std::max)(yminI, yminJ), 0.f) *
            (std::max)((std::min)(xmaxI, xmaxJ) - (std::max)(xminI, xminJ), 0.f);
    return intersection_area / (areaI + areaJ - intersection_area);
}

typedef struct {
    float score;
    int batch_index;
    int class_index;
    int box_index;
} filteredBoxes;

static void ref_nms(
        InferenceEngine::TBlob<float> &srcBoxes,
        InferenceEngine::TBlob<float> &srcScores,
        InferenceEngine::TBlob<int> &selected_idxs,
        nmsTF_test_params p
) {
    float *boxes = srcBoxes.data();
    float *scores = srcScores.data();

    InferenceEngine::SizeVector scores_dims = srcScores.getTensorDesc().getDims();
    int num_boxes = static_cast<int>(scores_dims[2]);
    int max_output_boxes_per_class = num_boxes;
    if (p.max_output_boxes_per_class.size())
        max_output_boxes_per_class = (std::min)(max_output_boxes_per_class, p.max_output_boxes_per_class[0]);

    float iou_threshold = 1.f;  //  Value range [0, 1]
    if (p.iou_threshold.size())
        iou_threshold = (std::min)(iou_threshold, p.iou_threshold[0]);

    float score_threshold = 0.f;
    if (p.score_threshold.size())
        score_threshold = p.score_threshold[0];

    int* selected_indices = selected_idxs.data();
    InferenceEngine::SizeVector selected_indices_dims = selected_idxs.getTensorDesc().getDims();

    InferenceEngine::SizeVector boxesStrides = srcBoxes.getTensorDesc().getBlockingDesc().getStrides();
    InferenceEngine::SizeVector scoresStrides = srcScores.getTensorDesc().getBlockingDesc().getStrides();

    // boxes shape: {num_batches, num_boxes, 4}
    // scores shape: {num_batches, num_classes, num_boxes}
    int num_batches = static_cast<int>(scores_dims[0]);
    int num_classes = static_cast<int>(scores_dims[1]);
    std::vector<filteredBoxes> fb;

    for (int batch = 0; batch < num_batches; batch++) {
        float *boxesPtr = boxes + batch * boxesStrides[0];
        for (int class_idx = 0; class_idx < num_classes; class_idx++) {
            float *scoresPtr = scores + batch * scoresStrides[0] + class_idx * scoresStrides[1];
            std::vector<std::pair<float, int> > scores_vector;
            for (int box_idx = 0; box_idx < num_boxes; box_idx++) {
                if (scoresPtr[box_idx] > score_threshold)
                    scores_vector.push_back(std::make_pair(scoresPtr[box_idx], box_idx));
            }

            if (scores_vector.size()) {
                std::sort(scores_vector.begin(), scores_vector.end(),
                          [](const std::pair<float, int>& l, const std::pair<float, int>& r) { return l.first > r.first; });

                int io_selection_size = 1;
                fb.push_back({ scores_vector[0].first, batch, class_idx, scores_vector[0].second });
                for (int box_idx = 1; (box_idx < static_cast<int>(scores_vector.size()) && io_selection_size < max_output_boxes_per_class); box_idx++) {
                    bool box_is_selected = true;
                    for (int idx = io_selection_size - 1; idx >= 0; idx--) {
                        float iou = intersectionOverUnion(&boxesPtr[scores_vector[box_idx].second * 4],
                                                          &boxesPtr[scores_vector[idx].second * 4], (p.center_point_box == 1));
                        if (iou > iou_threshold) {
                            box_is_selected = false;
                            break;
                        }
                    }

                    if (box_is_selected) {
                        scores_vector[io_selection_size] = scores_vector[box_idx];
                        io_selection_size++;
                        fb.push_back({ scores_vector[box_idx].first, batch, class_idx, scores_vector[box_idx].second });
                    }
                }
            }
        }
    }

    if(p.sort_result_descending)
        std::sort(fb.begin(), fb.end(), [](const filteredBoxes& l, const filteredBoxes& r) { return l.score > r.score; });
    int selected_indicesStride = selected_idxs.getTensorDesc().getBlockingDesc().getStrides()[0];
    int* selected_indicesPtr = selected_indices;
    size_t idx;
    for (idx = 0; idx < (std::min)(selected_indices_dims[0], fb.size()); idx++) {
        selected_indicesPtr[0] = fb[idx].batch_index;
        selected_indicesPtr[1] = fb[idx].class_index;
        selected_indicesPtr[2] = fb[idx].box_index;
        selected_indicesPtr += selected_indicesStride;
    }
    for (; idx < selected_indices_dims[0]; idx++) {
        selected_indicesPtr[0] = -1;
        selected_indicesPtr[1] = -1;
        selected_indicesPtr[2] = -1;
        selected_indicesPtr += selected_indicesStride;
    }
}

class MKLDNNCPUExtNonMaxSuppressionTFTests : public TestsCommon, public WithParamInterface<nmsTF_test_params> {
    std::string model_t2 = R"V0G0N(
<net Name="NonMaxSuppression_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputBoxes" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IBOXES_
                </port>
            </output>
        </layer>
        <layer name="InputScores" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">
                    _ISCORES_
                </port>
            </output>
        </layer>
        <layer name="non_max_suppression" type="NonMaxSuppression" precision="FP32" id="6">
            <data center_point_box="_CPB_" sort_result_descending="_SRD_"/>
            <input>
                <port id="1">
                    _IBOXES_
                </port>
                <port id="2">
                    _ISCORES_
                </port>
            </input>
            <output>
                <port id="6" precision="I32">
                    _IOUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="6" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="6" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string model_t3 = R"V0G0N(
<net Name="NonMaxSuppression_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputBoxes" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IBOXES_
                </port>
            </output>
        </layer>
        <layer name="InputScores" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">
                    _ISCORES_
                </port>
            </output>
        </layer>
        <layer name="InputBoxesPerClass" type="Input" precision="I32" id="3">
            <output>
                <port id="3"/>
            </output>
        </layer>
        <layer name="non_max_suppression" type="NonMaxSuppression" precision="FP32" id="6">
            <data center_point_box="_CPB_" sort_result_descending="_SRD_"/>
            <input>
                <port id="1">
                    _IBOXES_
                </port>
                <port id="2">
                    _ISCORES_
                </port>
                <port id="3" precision="I32"/>
            </input>
            <output>
                <port id="6" precision="I32">
                    _IOUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="6" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="6" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="6" to-port="3"/>
    </edges>
</net>
)V0G0N";
    std::string model_t4 = R"V0G0N(
<net Name="NonMaxSuppression_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputBoxes" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IBOXES_
                </port>
            </output>
        </layer>
        <layer name="InputScores" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">
                    _ISCORES_
                </port>
            </output>
        </layer>
        <layer name="InputBoxesPerClass" type="Input" precision="I32" id="3">
            <output>
                <port id="3"/>
            </output>
        </layer>
        <layer name="InputIouThr" type="Input" precision="FP32" id="4">
            <output>
                <port id="4"/>
            </output>
        </layer>
        <layer name="non_max_suppression" type="NonMaxSuppression" precision="FP32" id="6">
            <data center_point_box="_CPB_" sort_result_descending="_SRD_"/>
            <input>
                <port id="1">
                    _IBOXES_
                </port>
                <port id="2">
                    _ISCORES_
                </port>
                <port id="3" precision="I32"/>
                <port id="4"/>
            </input>
            <output>
                <port id="6" precision="I32">
                    _IOUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="6" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="6" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="6" to-port="3"/>
        <edge from-layer="4" from-port="4" to-layer="6" to-port="4"/>
    </edges>
</net>
)V0G0N";

    std::string model_t5 = R"V0G0N(
<net Name="NonMaxSuppression_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputBoxes" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IBOXES_
                </port>
            </output>
        </layer>
        <layer name="InputScores" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">
                    _ISCORES_
                </port>
            </output>
        </layer>
        <layer name="InputBoxesPerClass" type="Input" precision="I32" id="3">
            <output>
                <port id="3"/>
            </output>
        </layer>
        <layer name="InputIouThr" type="Input" precision="FP32" id="4">
            <output>
                <port id="4"/>
            </output>
        </layer>
        <layer name="InputScoreThr" type="Input" precision="FP32" id="5">
            <output>
                <port id="5"/>
            </output>
        </layer>
        <layer name="non_max_suppression" type="NonMaxSuppression" precision="FP32" id="6">
            <data center_point_box="_CPB_" sort_result_descending="_SRD_"/>
            <input>
                <port id="1">
                    _IBOXES_
                </port>
                <port id="2">
                    _ISCORES_
                </port>
                <port id="3" precision="I32"/>
                <port id="4"/>
                <port id="5"/>
            </input>
            <output>
                <port id="6" precision="I32">
                    _IOUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="6" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="6" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="6" to-port="3"/>
        <edge from-layer="4" from-port="4" to-layer="6" to-port="4"/>
        <edge from-layer="5" from-port="5" to-layer="6" to-port="5"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(nmsTF_test_params p) {
        std::string model;
        if (!p.max_output_boxes_per_class.size())
            model = model_t2;
        else if (!p.iou_threshold.size())
            model = model_t3;
        else if (!p.score_threshold.size())
            model = model_t4;
        else
            model = model_t5;

        std::string inBoxes;
        std::string inScores;
        std::string out;

        inBoxes += "<dim>" + std::to_string(p.scoresDim[0]) + "</dim>\n";
        inBoxes += "<dim>" + std::to_string(p.scoresDim[2]) + "</dim>\n";
        inBoxes += "<dim>4</dim>";


        for (auto& scr : p.scoresDim) {
            inScores += "<dim>";
            inScores += std::to_string(scr) + "</dim>\n";
        }

        out += "<dim>" + std::to_string(p.num_selected_indices) + "</dim>\n";
        out += "<dim>3</dim>";

        REPLACE_WITH_STR(model, "_IBOXES_", inBoxes);
        REPLACE_WITH_STR(model, "_ISCORES_", inScores);
        REPLACE_WITH_STR(model, "_IOUT_", out);
        REPLACE_WITH_NUM(model, "_CPB_", p.center_point_box);
        REPLACE_WITH_NUM(model, "_SRD_", p.sort_result_descending);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            nmsTF_test_params p = ::testing::WithParamInterface<nmsTF_test_params>::GetParam();
            std::string model = getModel(p);
            //std::cout << model << std::endl;
                        InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            //  Input
            InferenceEngine::BlobMap srcs;

            //  Input Boxes
            InferenceEngine::SizeVector boxesDim = {p.scoresDim[0], p.scoresDim[2], 4};
            InferenceEngine::Blob::Ptr srcBoxes = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, boxesDim, InferenceEngine::TensorDesc::getLayoutByDims(boxesDim) });
            srcBoxes->allocate();
            for (size_t i = 0; i < p.boxes.size(); i++) {
                static_cast<float*>(srcBoxes->buffer())[i] = static_cast<float>(p.boxes[i]);
            }
            //memcpy(srcBoxes->buffer(), &p.boxes[0], sizeof(float)*boxes.size());
            auto * srcBoxesPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(srcBoxes.get());
            if (srcBoxesPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputBoxes", srcBoxes));

            // Input Scores
            InferenceEngine::Blob::Ptr srcScores = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.scoresDim, InferenceEngine::TensorDesc::getLayoutByDims(p.scoresDim) });
            srcScores->allocate();
            for (size_t i = 0; i < p.scores.size(); i++) {
                static_cast<float*>(srcScores->buffer())[i] = static_cast<float>(p.scores[i]);
            }
            auto * srcScoresPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(srcScores.get());
            if (srcScoresPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputScores", srcScores));

            // Input BoxesPerClass
            InferenceEngine::Blob::Ptr srcBoxesPerClass;
            InferenceEngine::Blob::Ptr srcIouThr;
            InferenceEngine::Blob::Ptr srcScoreThr;
            if (p.max_output_boxes_per_class.size()) {
                srcBoxesPerClass = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, {}, InferenceEngine::TensorDesc::getLayoutByDims({}) });
                srcBoxesPerClass->allocate();
                memcpy(static_cast<int32_t*>(srcBoxesPerClass->buffer()), &p.max_output_boxes_per_class[0], sizeof(int32_t));
                auto * srcBoxesPerClassPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(srcBoxesPerClass.get());
                if (srcBoxesPerClassPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";
                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputBoxesPerClass", srcBoxesPerClass));
            }

            // Input IouThr
            if (p.iou_threshold.size()) {
                srcIouThr = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, {}, InferenceEngine::TensorDesc::getLayoutByDims({}) });
                srcIouThr->allocate();
                memcpy(static_cast<float*>(srcIouThr->buffer()), &p.iou_threshold[0], sizeof(float));
                auto * srcIouThrPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(srcIouThr.get());
                if (srcIouThrPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";
                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputIouThr", srcIouThr));
            }

            // Input ScoreThr
            if (p.score_threshold.size()) {
                srcScoreThr = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, {}, InferenceEngine::TensorDesc::getLayoutByDims({}) });
                srcScoreThr->allocate();
                memcpy(static_cast<float*>(srcScoreThr->buffer()), &p.score_threshold[0], sizeof(float));
                auto * srcScoreThrPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(srcScoreThr.get());
                if (srcScoreThrPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";
                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputScoreThr", srcScoreThr));
            }

            //  Output Data
            InferenceEngine::OutputsDataMap out;
            out = network.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;
            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();
            InferenceEngine::TBlob<int32_t>::Ptr output;
            output = InferenceEngine::make_shared_blob<int32_t>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            //  Infer
            graph.Infer(srcs, outputBlobs);

            // Output Reference
            if (!p.ref.size()) {
                InferenceEngine::TBlob <int32_t> selected_indices_ref(item.second->getTensorDesc());
                selected_indices_ref.allocate();
                ref_nms(*srcBoxesPtr, *srcScoresPtr, selected_indices_ref, p);
                compare(*output, selected_indices_ref);
            } else {
                //  Check results
                if (p.ref.size() != output->size())
                    FAIL() << "Wrong result vector size!";
                if (memcmp((*output).data(), &p.ref[0], output->byteSize()) != 0)
                    FAIL() << "Wrong result with compare TF reference!";
            }
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtNonMaxSuppressionTFTests, TestsNonMaxSuppression) {}

static std::vector<float> boxes = { 0.0, 0.0, 1.0, 1.0, 0.0, 0.1, 1.0, 1.1, 0.0, -0.1, 1.0, 0.9, 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0 };
static std::vector<float> scores = { 0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f };
static std::vector<int> reference = { 0,0,3,0,0,0,0,0,5 };

INSTANTIATE_TEST_CASE_P(
        TestsNonMaxSuppression, MKLDNNCPUExtNonMaxSuppressionTFTests,
        ::testing::Values(
// Params: center_point_box, sort_result_descending, scoresDim, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, num_selected_indices, ref

            nmsTF_test_params{ 1, 1, {1,1,6}, { 0.5f, 0.5f, 1.0f, 1.0f,0.5f, 0.6f, 1.0f, 1.0f,0.5f, 0.4f, 1.0f, 1.0f,0.5f, 10.5f, 1.0f, 1.0f, 0.5f, 10.6f, 1.0f, 1.0f, 0.5f, 100.5f, 1.0f, 1.0f },
            scores,{ 3 },{ 0.5f },{ 0.f }, 3, reference }, /*nonmaxsuppression_center_point_box_format*/

            nmsTF_test_params{ 0, 1, {1,1,6}, { 1.0, 1.0, 0.0, 0.0, 0.0, 0.1, 1.0, 1.1, 0.0, 0.9, 1.0, -0.1, 0.0, 10.0, 1.0, 11.0, 1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0 },
            scores,{ 3 },{ 0.5 },{ 0.0 }, 3, reference }, /*nonmaxsuppression_flipped_coordinates*/

            nmsTF_test_params{ 0, 1, { 1,1,10 },{ 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                               0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0 },
            { 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 },{ 3 },{ 0.5 },{ 0.0 }, 1,{ 0,0,0 } }, /*nonmaxsuppression_identical_boxes*/

            nmsTF_test_params{ 0, 1, { 1,1,6 }, boxes, scores,{ 2 },{ 0.5 },{ 0.0 }, 2,{ 0,0,3,0,0,0 } }, /*nonmaxsuppression_limit_output_size*/

            nmsTF_test_params{ 0, 1,{ 1,1,1 },{ 0.0, 0.0, 1.0, 1.0 }, { 0.9 },{ 3 },{ 0.5 },{ 0.0 }, 1, { 0,0,0 } }, /*nonmaxsuppression_single_box*/

            nmsTF_test_params{ 0, 1, { 1,1,6 }, boxes, scores, { 3 }, { 0.5 }, { 0.0 }, 3, reference }, /*nonmaxsuppression_suppress_by_IOU*/

            nmsTF_test_params{ 0, 1, { 1,1,6 }, boxes, scores, { 3 }, { 0.5 }, { 0.4 }, 2, { 0,0,3,0,0,0 } }, /*nonmaxsuppression_suppress_by_IOU_and_scores*/

            nmsTF_test_params{ 0, 0, { 2,1,6 },{ 0.0, 0.0, 1.0, 1.0, 0.0, 0.1, 1.0, 1.1, 0.0, -0.1, 1.0, 0.9, 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                                             0.0, 0.0, 1.0, 1.0, 0.0, 0.1, 1.0, 1.1, 0.0, -0.1, 1.0, 0.9, 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0 },
            { 0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3 },{ 2 },{ 0.5 },{ 0.0 }, 4,{ 0,0,3,0,0,0,1,0,3,1,0,0 } }, /*nonmaxsuppression_two_batches*/

            nmsTF_test_params{ 0, 1, { 2,1,6 },{ 0.0, 0.0, 1.0, 1.0, 0.0, 0.1, 1.0, 1.1, 0.0, -0.1, 1.0, 0.9, 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                                             0.0, 0.0, 1.0, 1.0, 0.0, 0.1, 1.0, 1.1, 0.0, -0.1, 1.0, 0.9, 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0 },
            { 0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3 },{ 2 },{ 0.5 },{ 0.0 }, 4,{ 0,0,3,1,0,3,0,0,0,1,0,0 } }, /*nonmaxsuppression_two_batches*/

            nmsTF_test_params{ 0, 0, { 1,2,6 }, boxes,
            { 0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3 },{ 2 },{ 0.5 },{ 0.0 }, 4,{ 0,0,3,0,0,0,0,1,3,0,1,0 } }, /*nonmaxsuppression_two_classes*/

            nmsTF_test_params{ 0, 1, { 1,2,6 }, boxes,
            { 0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3 },{ 2 },{ 0.5 },{ 0.0 }, 4,{ 0,0,3,0,1,3,0,0,0,0,1,0 } }, /*nonmaxsuppression_two_classes*/

            nmsTF_test_params{ 0, 1, { 1,1,6 }, boxes, scores, { 3 }, { 0.5 }, {}, 3, reference }, /*nonmaxsuppression_no_score_threshold*/

            nmsTF_test_params{ 0, 1, { 1,1,6 }, boxes, scores, { 3 }, {}, {}, 3, { 0,0,3,0,0,0,0,0,1 } }, /*nonmaxsuppression_no_iou_threshold_and_score_threshold*/

            nmsTF_test_params{ 0, 1, { 1,1,6 }, boxes, scores, {}, {}, {}, 3, {} } /*nonmaxsuppression_no_max_output_boxes_per_class_and_iou_threshold_and_score_threshold*/
));
