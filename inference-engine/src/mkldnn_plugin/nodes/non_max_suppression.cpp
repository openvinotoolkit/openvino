// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include <utility>
#include <queue>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class NonMaxSuppressionImpl: public ExtLayerBase {
public:
    explicit NonMaxSuppressionImpl(const CNNLayer* layer) {
        try {
            logPrefix = "NMS layer with name '" + layer->name + "' ";
            if (layer->insData.size() < 2 || layer->insData.size() > 6)
                THROW_IE_EXCEPTION << logPrefix << "has incorrect number of input edges: " << layer->insData.size();

            if (layer->outData.size() < 1 || layer->outData.size() > 3)
                THROW_IE_EXCEPTION << logPrefix << "has incorrect number of output edges: " << layer->outData.size();

            // TODO: remove legacy attribute presentation after migration on opset1
            if (layer->CheckParamPresence("center_point_box")) {
                bool center_point_box = layer->GetParamAsBool("center_point_box", false);
                boxEncodingType = center_point_box ? boxEncoding::CENTER : boxEncoding::CORNER;
            } else if (layer->CheckParamPresence("box_encoding")) {
                std::string boxEncAttr = layer->GetParamAsString("box_encoding", "corner");
                if (boxEncAttr == "corner") {
                    boxEncodingType = boxEncoding::CORNER;
                } else if (boxEncAttr == "center") {
                    boxEncodingType = boxEncoding::CENTER;
                } else {
                    THROW_IE_EXCEPTION << logPrefix << "has unsupported 'box_encoding' attribute: " << boxEncAttr;
                }
            }

            sort_result_descending = layer->GetParamAsBool("sort_result_descending", true);

            const std::vector<Precision> supportedFloatPrecision = {Precision::FP32, Precision::BF16};
            const std::vector<Precision> supportedIntOutputPrecision = {Precision::I32, Precision::I64};

            auto boxesDataPtr = layer->insData[NMS_BOXES].lock();
            if (boxesDataPtr == nullptr) {
                THROW_IE_EXCEPTION << logPrefix << "has nullable 'boxes' input";
            }
            checkPrecision(boxesDataPtr, supportedFloatPrecision, "boxes", inType);
            const SizeVector &boxes_dims = boxesDataPtr->getTensorDesc().getDims();
            num_batches = boxes_dims[0];
            num_boxes = boxes_dims[1];
            if (boxes_dims.size() != 3)
                THROW_IE_EXCEPTION << logPrefix << "has unsupported 'boxes' input rank: " << boxes_dims.size();
            if (boxes_dims[2] != 4)
                THROW_IE_EXCEPTION << logPrefix << "has unsupported 'boxes' input 3rd dimension size: " << boxes_dims[2];


            auto scoresDataPtr = layer->insData[NMS_SCORES].lock();
            if (scoresDataPtr == nullptr) {
                THROW_IE_EXCEPTION << logPrefix << "has nullable 'scores' input";
            }
            checkPrecision(scoresDataPtr, supportedFloatPrecision, "scores", inType);
            const SizeVector &scores_dims = scoresDataPtr->getTensorDesc().getDims();
            num_classes = scores_dims[1];
            if (scores_dims.size() != 3)
                THROW_IE_EXCEPTION << logPrefix << "has unsupported 'scores' input rank: " << scores_dims.size();

            if (num_batches != scores_dims[0])
                THROW_IE_EXCEPTION << logPrefix << " num_batches is different in 'boxes' and 'scores' inputs";
            if (num_boxes != scores_dims[2])
                THROW_IE_EXCEPTION << logPrefix << " num_boxes is different in 'boxes' and 'scores' inputs";

            numFiltBox.resize(num_batches);
            for (size_t i = 0; i < numFiltBox.size(); i++)
                numFiltBox[i].resize(num_classes);

            if (layer->insData.size() > NMS_MAXOUTPUTBOXESPERCLASS) {
                const std::vector<Precision> supportedPrecision = {Precision::I16, Precision::U8, Precision::I8, Precision::U16, Precision::I32,
                                                                   Precision::U32, Precision::I64, Precision::U64};
                check1DInput(layer->insData[NMS_MAXOUTPUTBOXESPERCLASS], supportedPrecision, "max_output_boxes_per_class");
            }

            if (layer->insData.size() > NMS_IOUTHRESHOLD) {
                check1DInput(layer->insData[NMS_IOUTHRESHOLD], supportedFloatPrecision, "iou_threshold");
            }

            if (layer->insData.size() > NMS_SCORETHRESHOLD) {
                check1DInput(layer->insData[NMS_SCORETHRESHOLD], supportedFloatPrecision, "score_threshold");
            }

            if (layer->insData.size() > NMS_SOFTNMSSIGMA) {
                check1DInput(layer->insData[NMS_SOFTNMSSIGMA], supportedFloatPrecision, "soft_nms_sigma");
            }

            checkOutput(layer->outData[NMS_SELECTEDINDICES], supportedIntOutputPrecision, "selected_indices");

            if (layer->outData.size() > NMS_SELECTEDSCORES) {
                checkOutput(layer->outData[NMS_SELECTEDSCORES], supportedFloatPrecision, "selected_scores");
            }

            if (layer->outData.size() > NMS_VALIDOUTPUTS) {
                checkPrecision(layer->outData[NMS_VALIDOUTPUTS], supportedIntOutputPrecision, "valid_outputs", outType);
                const SizeVector &valid_outputs_dims = layer->outData[NMS_VALIDOUTPUTS]->getTensorDesc().getDims();
                if (valid_outputs_dims.size() != 1)
                    THROW_IE_EXCEPTION << logPrefix << "has unsupported 'valid_outputs' output rank: " << valid_outputs_dims.size();
                if (valid_outputs_dims[0] != 1)
                    THROW_IE_EXCEPTION << logPrefix << "has unsupported 'valid_outputs' output 1st dimension size: " << valid_outputs_dims[1];
            }

            LayerConfig config;
            for (size_t i = 0; i < layer->insData.size(); i++) {
                DataConfig inConfig;

                Precision inPrecision = i == NMS_MAXOUTPUTBOXESPERCLASS ? Precision::I32 : Precision::FP32;
                auto validDataPtr = layer->insData[i].lock();
                if (validDataPtr == nullptr) {
                    THROW_IE_EXCEPTION << logPrefix << "has nullable " << i << "th input";
                }
                const SizeVector& inDims = validDataPtr->getTensorDesc().getDims();
                inConfig.desc = TensorDesc(inPrecision, inDims, InferenceEngine::TensorDesc::getLayoutByDims(inDims));
                config.inConfs.push_back(inConfig);
            }
            for (size_t i = 0; i < layer->outData.size(); i++) {
                DataConfig outConfig;

                Precision outPrecision = i == NMS_SELECTEDSCORES ? Precision::FP32 : Precision::I32;
                const SizeVector& outDims = layer->outData[i]->getTensorDesc().getDims();
                outConfig.desc = TensorDesc(outPrecision, outDims, InferenceEngine::TensorDesc::getLayoutByDims(outDims));
                config.outConfs.push_back(outConfig);
            }

            config.dynBatchSupport = false;
            confs.push_back(config);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    float intersectionOverUnion(const float *boxesI, const float *boxesJ) {
        float yminI, xminI, ymaxI, xmaxI, yminJ, xminJ, ymaxJ, xmaxJ;
        if (boxEncodingType == boxEncoding::CENTER) {
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

    struct filteredBoxes {
        float score;
        int batch_index;
        int class_index;
        int box_index;
        filteredBoxes() {}
        filteredBoxes(float _score, int _batch_index, int _class_index, int _box_index) :
                      score(_score), batch_index(_batch_index), class_index(_class_index), box_index(_box_index) {}
    };

    struct boxInfo {
        float score;
        int idx;
        int suppress_begin_index;
    };

    void nmsWithSoftSigma(const float *boxes, const float *scores, const SizeVector &boxesStrides, const SizeVector &scoresStrides,
                          std::vector<filteredBoxes> &filtBoxes) {
        auto less = [](const boxInfo& l, const boxInfo& r) {
            return l.score < r.score || ((l.score == r.score) && (l.idx > r.idx));
        };

        auto coeff = [&](float iou) {
            const float weight = std::exp(scale * iou * iou);
            return iou <= iou_threshold ? weight : 0.0f;
        };

        parallel_for2d(num_batches, num_classes, [&](int batch_idx, int class_idx) {
            std::vector<filteredBoxes> fb;
            const float *boxesPtr = boxes + batch_idx * boxesStrides[0];
            const float *scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

            std::priority_queue<boxInfo, std::vector<boxInfo>, decltype(less)> sorted_boxes(less);
            for (int box_idx = 0; box_idx < num_boxes; box_idx++) {
                if (scoresPtr[box_idx] > score_threshold)
                    sorted_boxes.emplace(boxInfo({scoresPtr[box_idx], box_idx, 0}));
            }

            fb.reserve(sorted_boxes.size());
            if (sorted_boxes.size() > 0) {
                while (fb.size() < max_output_boxes_per_class && !sorted_boxes.empty()) {
                    boxInfo currBox = sorted_boxes.top();
                    float origScore = currBox.score;
                    sorted_boxes.pop();

                    bool box_is_selected = true;
                    for (int idx = static_cast<int>(fb.size()) - 1; idx >= currBox.suppress_begin_index; idx--) {
                        float iou = intersectionOverUnion(&boxesPtr[currBox.idx * 4], &boxesPtr[fb[idx].box_index * 4]);
                        currBox.score *= coeff(iou);
                        if (iou >= iou_threshold) {
                            box_is_selected = false;
                            break;
                        }
                        if (currBox.score <= score_threshold)
                            break;
                    }

                    currBox.suppress_begin_index = fb.size();
                    if (box_is_selected) {
                        if (currBox.score == origScore) {
                            fb.push_back({ currBox.score, batch_idx, class_idx, currBox.idx });
                            continue;
                        }
                        if (currBox.score > score_threshold) {
                            sorted_boxes.push(currBox);
                        }
                    }
                }
            }
            numFiltBox[batch_idx][class_idx] = fb.size();
            size_t offset = batch_idx*num_classes*max_output_boxes_per_class + class_idx*max_output_boxes_per_class;
            cpu_memcpy(filtBoxes.data() + offset, fb.data(), fb.size() * sizeof(filteredBoxes));
        });
    }

    void nmsWithoutSoftSigma(const float *boxes, const float *scores, const SizeVector &boxesStrides, const SizeVector &scoresStrides,
                             std::vector<filteredBoxes> &filtBoxes) {
        int max_out_box = static_cast<int>(max_output_boxes_per_class);
        parallel_for2d(num_batches, num_classes, [&](int batch_idx, int class_idx) {
            const float *boxesPtr = boxes + batch_idx * boxesStrides[0];
            const float *scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

            std::vector<std::pair<float, int>> sorted_boxes;
            for (int box_idx = 0; box_idx < num_boxes; box_idx++) {
                if (scoresPtr[box_idx] > score_threshold)
                    sorted_boxes.emplace_back(std::make_pair(scoresPtr[box_idx], box_idx));
            }

            int io_selection_size = 0;
            if (sorted_boxes.size() > 0) {
                parallel_sort(sorted_boxes.begin(), sorted_boxes.end(),
                              [](const std::pair<float, int>& l, const std::pair<float, int>& r) {
                                    return (l.first > r.first || ((l.first == r.first) && (l.second < r.second)));
                                });
                size_t offset = batch_idx*num_classes*max_output_boxes_per_class + class_idx*max_output_boxes_per_class;
                filteredBoxes *fb = filtBoxes.data() + offset;
                fb[0] = filteredBoxes(sorted_boxes[0].first, batch_idx, class_idx, sorted_boxes[0].second);
                io_selection_size++;
                for (size_t box_idx = 1; (box_idx < sorted_boxes.size()) && (io_selection_size < max_out_box); box_idx++) {
                    bool box_is_selected = true;
                    for (int idx = io_selection_size - 1; idx >= 0; idx--) {
                        float iou = intersectionOverUnion(&boxesPtr[sorted_boxes[box_idx].second * 4], &boxesPtr[fb[idx].box_index * 4]);
                        if (iou >= iou_threshold) {
                            box_is_selected = false;
                            break;
                        }
                    }

                    if (box_is_selected) {
                        fb[io_selection_size] = filteredBoxes(sorted_boxes[box_idx].first, batch_idx, class_idx, sorted_boxes[box_idx].second);
                        io_selection_size++;
                    }
                }
            }
            numFiltBox[batch_idx][class_idx] = io_selection_size;
        });
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        const float *boxes = inputs[NMS_BOXES]->cbuffer().as<const float *>() + inputs[NMS_BOXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const float *scores = inputs[NMS_SCORES]->cbuffer().as<const float *>() + inputs[NMS_SCORES]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        max_output_boxes_per_class = outputs.size() > NMS_SELECTEDSCORES ? 0 : num_boxes;
        if (inputs.size() > NMS_MAXOUTPUTBOXESPERCLASS) {
            max_output_boxes_per_class = (inputs[NMS_MAXOUTPUTBOXESPERCLASS]->cbuffer().as<int *>() +
                                          inputs[NMS_MAXOUTPUTBOXESPERCLASS]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];
        }

        if (max_output_boxes_per_class == 0)
            return OK;

        iou_threshold = outputs.size() > NMS_SELECTEDSCORES ? 0.0f : 1.0f;
        if (inputs.size() > NMS_IOUTHRESHOLD)
            iou_threshold = (inputs[NMS_IOUTHRESHOLD]->cbuffer().as<float *>() +
                             inputs[NMS_IOUTHRESHOLD]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];

        score_threshold = 0.0f;
        if (inputs.size() > NMS_SCORETHRESHOLD)
            score_threshold = (inputs[NMS_SCORETHRESHOLD]->cbuffer().as<float *>() +
                               inputs[NMS_SCORETHRESHOLD]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];

        soft_nms_sigma = 0.0f;
        if (inputs.size() > NMS_SOFTNMSSIGMA)
            soft_nms_sigma = (inputs[NMS_SOFTNMSSIGMA]->cbuffer().as<float *>() +
                              inputs[NMS_SOFTNMSSIGMA]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];
        scale = 0.0f;
        if (soft_nms_sigma > 0.0) {
            scale = -0.5 / soft_nms_sigma;
        }

        int *selected_indices = outputs[NMS_SELECTEDINDICES]->buffer().as<int *>() +
                                outputs[NMS_SELECTEDINDICES]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        float *selected_scores = nullptr;
        if (outputs.size() > NMS_SELECTEDSCORES)
            selected_scores = outputs[NMS_SELECTEDSCORES]->buffer().as<float *>() +
                              outputs[NMS_SELECTEDSCORES]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        int *valid_outputs = nullptr;
        if (outputs.size() > NMS_VALIDOUTPUTS)
            valid_outputs = outputs[NMS_VALIDOUTPUTS]->buffer().as<int *>() +
                            outputs[NMS_VALIDOUTPUTS]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const SizeVector &boxesStrides = inputs[NMS_BOXES]->getTensorDesc().getBlockingDesc().getStrides();
        const SizeVector &scoresStrides = inputs[NMS_SCORES]->getTensorDesc().getBlockingDesc().getStrides();

        std::vector<filteredBoxes> filtBoxes(max_output_boxes_per_class * num_batches * num_classes);

        if (soft_nms_sigma == 0.0f) {
            nmsWithoutSoftSigma(boxes, scores, boxesStrides, scoresStrides, filtBoxes);
        } else {
            nmsWithSoftSigma(boxes, scores, boxesStrides, scoresStrides, filtBoxes);
        }

        size_t startOffset = numFiltBox[0][0];
        for (size_t b = 0; b < numFiltBox.size(); b++) {
            size_t batchOffset = b*num_classes*max_output_boxes_per_class;
            for (size_t c = (b == 0 ? 1 : 0); c < numFiltBox[b].size(); c++) {
                size_t offset = batchOffset + c*max_output_boxes_per_class;
                cpu_memcpy(filtBoxes.data() + startOffset, filtBoxes.data() + offset,
                           numFiltBox[b][c] * sizeof(filteredBoxes));
                startOffset += numFiltBox[b][c];
            }
        }
        filtBoxes.resize(startOffset);

        // need more particular comparator to get deterministic behaviour
        // escape situation when filtred boxes with same score have different position from launch to launch
        if (sort_result_descending) {
            parallel_sort(filtBoxes.begin(), filtBoxes.end(),
                          [](const filteredBoxes& l, const filteredBoxes& r) {
                                return (l.score > r.score) ||
                                (l.score ==  r.score && l.batch_index < r.batch_index) ||
                                (l.score ==  r.score && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
                                (l.score ==  r.score && l.batch_index == r.batch_index && l.class_index == r.class_index && l.box_index < r.box_index);
                            });
        }

        size_t validOutputs = std::min(filtBoxes.size(), static_cast<size_t>(outputs[NMS_SELECTEDINDICES]->getTensorDesc().getDims()[0]));

        int selectedIndicesStride = outputs[NMS_SELECTEDINDICES]->getTensorDesc().getBlockingDesc().getStrides()[0];
        int *selectedIndicesPtr = selected_indices;
        float *selectedScoresPtr = selected_scores;

        for (size_t idx = 0; idx < validOutputs; idx++) {
            selectedIndicesPtr[0] = filtBoxes[idx].batch_index;
            selectedIndicesPtr[1] = filtBoxes[idx].class_index;
            selectedIndicesPtr[2] = filtBoxes[idx].box_index;
            selectedIndicesPtr += selectedIndicesStride;
            if (outputs.size() > NMS_SELECTEDSCORES) {
                selectedScoresPtr[0] = static_cast<float>(filtBoxes[idx].batch_index);
                selectedScoresPtr[1] = static_cast<float>(filtBoxes[idx].class_index);
                selectedScoresPtr[2] = static_cast<float>(filtBoxes[idx].score);
                selectedScoresPtr += selectedIndicesStride;
            }
        }
        if (outputs.size() > NMS_VALIDOUTPUTS)
            *valid_outputs = static_cast<int>(validOutputs);

        return OK;
    }

private:
    // input
    const size_t NMS_BOXES = 0;
    const size_t NMS_SCORES = 1;
    const size_t NMS_MAXOUTPUTBOXESPERCLASS = 2;
    const size_t NMS_IOUTHRESHOLD = 3;
    const size_t NMS_SCORETHRESHOLD = 4;
    const size_t NMS_SOFTNMSSIGMA = 5;

    // output
    const size_t NMS_SELECTEDINDICES = 0;
    const size_t NMS_SELECTEDSCORES = 1;
    const size_t NMS_VALIDOUTPUTS = 2;

    enum class boxEncoding {
        CORNER,
        CENTER
    };
    boxEncoding boxEncodingType = boxEncoding::CORNER;
    bool sort_result_descending = true;

    size_t num_batches;
    size_t num_boxes;
    size_t num_classes;

    size_t max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    float scale;

    std::vector<std::vector<size_t>> numFiltBox;
    const std::string inType = "input", outType = "output";
    std::string logPrefix;

    void checkPrecision(const DataPtr &dataPtr, const std::vector<Precision> precList, const std::string name, const std::string type) {
        const TensorDesc &tensorDesc = dataPtr->getTensorDesc();
        if (std::find(precList.begin(), precList.end(), tensorDesc.getPrecision()) == precList.end())
            THROW_IE_EXCEPTION << logPrefix << " has unsupported '" << name << "' " << type << " precision: " << tensorDesc.getPrecision();
    }

    void check1DInput(const DataWeakPtr &dataPtr, const std::vector<Precision> precList, const std::string name) {
        auto lockDataPtr = dataPtr.lock();
        if (lockDataPtr == nullptr) {
            THROW_IE_EXCEPTION << logPrefix << "has nullable '" << name << "' input";
        }

        checkPrecision(lockDataPtr, precList, name, inType);

        const SizeVector &dims = lockDataPtr->getTensorDesc().getDims();
        if (dims.size() != 0 && dims.size() != 1)
            THROW_IE_EXCEPTION << logPrefix << "has unsupported '" << name << "' input rank: " << dims.size();
        if (dims.size() == 1)
            if (dims[0] != 1)
                THROW_IE_EXCEPTION << logPrefix << "has unsupported '" << name << "' input 1st dimension size: " << dims[0];
    }

    void checkOutput(const DataPtr &dataPtr, const std::vector<Precision> precList, const std::string name) {
        checkPrecision(dataPtr, precList, name, outType);

        const SizeVector &dims = dataPtr->getTensorDesc().getDims();
        if (dims.size() != 2)
            THROW_IE_EXCEPTION << logPrefix << "has unsupported '" << name << "' output rank: " << dims.size();
        if (dims[1] != 3)
            THROW_IE_EXCEPTION << logPrefix << "has unsupported '" << name << "' output 2nd dimension size: " << dims[1];
    }
};

REG_FACTORY_FOR(NonMaxSuppressionImpl, NonMaxSuppression);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
