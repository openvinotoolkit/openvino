// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <map>
#include <tuple>

#include "ngraph/op/detection_output.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

enum {
    idxLocation,
    idxConfidence,
    idxPriors,
    idxArmConfidence,
    idxArmLocation,
    numInputs
};

class referenceDetectionOutput {
  private:
    struct NormalizedBBox {
        float xmin = 0.0f;
        float ymin = 0.0f;
        float xmax = 0.0f;
        float ymax = 0.0f;
        float size = 0;
    };
    using LabelBBox = std::map<int, std::vector<NormalizedBBox>>;

    ngraph::op::DetectionOutputAttrs attrs;
    size_t numImages;
    size_t priorSize;
    size_t numPriors;
    size_t numLocClasses;
    size_t offset;

    void GetLocPredictions(const float* locData, std::vector<LabelBBox>& locations) {
        locations.resize(numImages);
        for (size_t i = 0; i < numImages; ++i) {
            LabelBBox& labelBbox = locations[i];
            for (size_t p = 0; p < numPriors; ++p) {
                size_t startIdx = p * numLocClasses * 4;
                for (size_t c = 0; c < numLocClasses; ++c) {
                    int label = attrs.share_location ? -1 : c;
                    if (labelBbox.find(label) == labelBbox.end()) {
                      labelBbox[label].resize(numPriors);
                    }
                    labelBbox[label][p].xmin = locData[startIdx + c * 4];
                    labelBbox[label][p].ymin = locData[startIdx + c * 4 + 1];
                    labelBbox[label][p].xmax = locData[startIdx + c * 4 + 2];
                    labelBbox[label][p].ymax = locData[startIdx + c * 4 + 3];
                }
            }
            locData += numPriors * numLocClasses * 4;
        }
    }

    void GetConfidenceScores(const float* confData, std::vector<std::map<int, std::vector<float>>>& confPreds) {
        confPreds.resize(numImages);
        for (int i = 0; i < numImages; ++i) {
            std::map<int, std::vector<float>>& labelScores = confPreds[i];
            for (int p = 0; p < numPriors; ++p) {
                int startIdx = p * attrs.num_classes;
                for (int c = 0; c < attrs.num_classes; ++c) {
                    labelScores[c].push_back(confData[startIdx + c]);
                }
            }
            confData += numPriors * attrs.num_classes;
        }
    }

    void OSGetConfidenceScores(const float* confData, const float* armConfData, std::vector<std::map<int, std::vector<float>>>& confPreds) {
        confPreds.resize(numImages);
        for (int i = 0; i < numImages; ++i) {
            std::map<int, std::vector<float>>& labelScores = confPreds[i];
            for (int p = 0; p < numPriors; ++p) {
                int startIdx = p * attrs.num_classes;
                if (armConfData[p * 2 + 1] < attrs.objectness_score) {
                    for (int c = 0; c < attrs.num_classes; ++c) {
                        c == attrs.background_label_id ? labelScores[c].push_back(1.0f) : labelScores[c].push_back(0.0f);
                    }
                } else {
                    for (int c = 0; c < attrs.num_classes; ++c) {
                        labelScores[c].push_back(confData[startIdx + c]);
                    }
                }
            }
            confData += numPriors * attrs.num_classes;
            armConfData += numPriors * 2;
        }
    }

    float BBoxSize(const NormalizedBBox& bbox) {
        if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin) {
            return 0;
        } else {
            float width = bbox.xmax - bbox.xmin;
            float height = bbox.ymax - bbox.ymin;
            return width * height;
        }
    }

    void GetPriorBBoxes(const float* priorData, std::vector<std::vector<NormalizedBBox>>& priorBboxes,
                        std::vector<std::vector<std::vector<float>>>& priorVariances) {
        priorBboxes.resize(numImages);
        priorVariances.resize(numImages);
        for (int n = 0; n < numImages; n++) {
            priorData += attrs.variance_encoded_in_target ? n*numPriors*priorSize : 2*n*numPriors*priorSize;
            std::vector<NormalizedBBox>& currPrBbox = priorBboxes[n];
            std::vector<std::vector<float>>& currPrVar = priorVariances[n];
            for (int i = 0; i < numPriors; ++i) {
                int start_idx = i * priorSize;
                NormalizedBBox bbox;
                bbox.xmin = priorData[start_idx + 0 + offset];
                bbox.ymin = priorData[start_idx + 1 + offset];
                bbox.xmax = priorData[start_idx + 2 + offset];
                bbox.ymax = priorData[start_idx + 3 + offset];
                float bbox_size = BBoxSize(bbox);
                bbox.size = bbox_size;
                currPrBbox.push_back(bbox);
            }
            if (!attrs.variance_encoded_in_target) {
                const float *priorVar = priorData + numPriors*priorSize;
                for (int i = 0; i < numPriors; ++i) {
                    int start_idx = i * 4;
                    std::vector<float> var;
                    for (int j = 0; j < 4; ++j) {
                        var.push_back(priorVar[start_idx + j]);
                    }
                    currPrVar.push_back(var);
                }
            }
        }
    }

    void DecodeBBox(const NormalizedBBox& priorBboxes, const std::vector<float>& priorVariances, const NormalizedBBox& bbox, NormalizedBBox& decodeBbox) {
        float priorXmin = priorBboxes.xmin;
        float priorYmin = priorBboxes.ymin;
        float priorXmax = priorBboxes.xmax;
        float priorYmax = priorBboxes.ymax;

        if (!attrs.normalized) {
            priorXmin /= attrs.input_width;
            priorYmin /= attrs.input_height;
            priorXmax /= attrs.input_width;
            priorYmax /= attrs.input_height;
        }
        if (attrs.code_type == "caffe.PriorBoxParameter.CORNER") {
            if (attrs.variance_encoded_in_target) {
                decodeBbox.xmin = priorXmin + bbox.xmin;
                decodeBbox.ymin = priorYmin + bbox.ymin;
                decodeBbox.xmax = priorXmax + bbox.xmax;
                decodeBbox.ymax = priorYmax + bbox.ymax;
            } else {
                decodeBbox.xmin = priorXmin + priorVariances[0] * bbox.xmin;
                decodeBbox.ymin = priorYmin + priorVariances[1] * bbox.ymin;
                decodeBbox.xmax = priorXmax + priorVariances[2] * bbox.xmax;
                decodeBbox.ymax = priorYmax + priorVariances[3] * bbox.ymax;
            }
        } else if (attrs.code_type == "caffe.PriorBoxParameter.CENTER_SIZE") {
            float priorWidth = priorXmax - priorXmin;
            float priorHeight = priorYmax - priorYmin;
            float priorCenterX = (priorXmin + priorXmax) / 2.0;
            float priorCenterY = (priorYmin + priorYmax) / 2.0;
            float decodeBboxCenterX, decodeBboxCenterY;
            float decodeBboxWidth, decodeBboxHeight;
            if (attrs.variance_encoded_in_target) {
                decodeBboxCenterX = bbox.xmin * priorWidth + priorCenterX;
                decodeBboxCenterY = bbox.ymin * priorHeight + priorCenterY;
                decodeBboxWidth = exp(bbox.xmax) * priorWidth;
                decodeBboxHeight = exp(bbox.ymax) * priorHeight;
            } else {
                decodeBboxCenterX = priorVariances[0] * bbox.xmin * priorWidth + priorCenterX;
                decodeBboxCenterY = priorVariances[1] * bbox.ymin * priorHeight + priorCenterY;
                decodeBboxWidth = exp(priorVariances[2] * bbox.xmax) * priorWidth;
                decodeBboxHeight = exp(priorVariances[3] * bbox.ymax) * priorHeight;
            }
            decodeBbox.xmin = decodeBboxCenterX - decodeBboxWidth / 2.0;
            decodeBbox.ymin = decodeBboxCenterY - decodeBboxHeight / 2.0;
            decodeBbox.xmax = decodeBboxCenterX + decodeBboxWidth / 2.0;
            decodeBbox.ymax = decodeBboxCenterY + decodeBboxHeight / 2.0;
        }
        if (attrs.clip_before_nms) {
            decodeBbox.xmin = (std::max)(0.0f, (std::min)(1.0f, decodeBbox.xmin));
            decodeBbox.ymin = (std::max)(0.0f, (std::min)(1.0f, decodeBbox.ymin));
            decodeBbox.xmax = (std::max)(0.0f, (std::min)(1.0f, decodeBbox.xmax));
            decodeBbox.ymax = (std::max)(0.0f, (std::min)(1.0f, decodeBbox.ymax));
        }
        float bboxSize = BBoxSize(decodeBbox);
        decodeBbox.size = bboxSize;
    }

    void DecodeBBoxes(const std::vector<NormalizedBBox>& priorBboxes, const std::vector<std::vector<float>>& priorVariances,
                      const std::vector<NormalizedBBox>& labelLocPreds, std::vector<NormalizedBBox>& decodeBboxes) {
        int numBboxes = priorBboxes.size();
        for (int i = 0; i < numBboxes; ++i) {
          NormalizedBBox decodeBbox;
          DecodeBBox(priorBboxes[i], priorVariances[i], labelLocPreds[i], decodeBbox);
          decodeBboxes.push_back(decodeBbox);
        }
    }

    void DecodeBBoxesAll(const std::vector<LabelBBox>& locPreds, const std::vector<std::vector<NormalizedBBox>>& priorBboxes,
                         const std::vector<std::vector<std::vector<float>>>& priorVariances, std::vector<LabelBBox>& decodeBboxes) {
        decodeBboxes.resize(numImages);
        for (int i = 0; i < numImages; ++i) {
            LabelBBox& decodeBboxesImage = decodeBboxes[i];
            const std::vector<NormalizedBBox>& currPrBbox = priorBboxes[i];
            const std::vector<std::vector<float>>& currPrVar = priorVariances[i];
            for (int c = 0; c < numLocClasses; ++c) {
                int label = attrs.share_location ? -1 : c;
                if (label == attrs.background_label_id) {
                    continue;
                }
                const std::vector<NormalizedBBox>& labelLocPreds = locPreds[i].find(label)->second;
                DecodeBBoxes(currPrBbox, currPrVar, labelLocPreds, decodeBboxesImage[label]);
            }
        }
    }

    void CasRegDecodeBBoxesAll(const std::vector<LabelBBox>& locPreds, const std::vector<std::vector<NormalizedBBox>>& priorBboxes,
                               const std::vector<std::vector<std::vector<float>>>& priorVariances, std::vector<LabelBBox>& decodeBboxes,
                               const std::vector<LabelBBox>& armLocPreds) {
        decodeBboxes.resize(numImages);
        for (int i = 0; i < numImages; ++i) {
            LabelBBox& decodeBboxesImage = decodeBboxes[i];
            const std::vector<NormalizedBBox>& currPrBbox = priorBboxes[i];
            const std::vector<std::vector<float>>& currPrVar = priorVariances[i];
            for (int c = 0; c < numLocClasses; ++c) {
                int label = attrs.share_location ? -1 : c;
                if (label == attrs.background_label_id) {
                    continue;
                }
                const std::vector<NormalizedBBox>& labelArmLocPreds = armLocPreds[i].find(label)->second;
                std::vector<NormalizedBBox> decodePriorBboxes;
                DecodeBBoxes(currPrBbox, currPrVar, labelArmLocPreds, decodePriorBboxes);
                const std::vector<NormalizedBBox>& labelLocPreds = locPreds[i].find(label)->second;
                DecodeBBoxes(decodePriorBboxes, currPrVar, labelLocPreds, decodeBboxesImage[label]);
            }
        }
    }

    template <typename T>
    static bool SortScorePairDescend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2) {
        return pair1.first > pair2.first;
    }

    void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold,
                          const int topK, std::vector<std::pair<float, int>>& scoreIndexVec) {
        for (int i = 0; i < scores.size(); ++i) {
            if (scores[i] > threshold) {
                scoreIndexVec.push_back(std::make_pair(scores[i], i));
            }
        }

        std::stable_sort(scoreIndexVec.begin(), scoreIndexVec.end(), SortScorePairDescend<int>);
        if (topK > -1 && topK < scoreIndexVec.size()) {
          scoreIndexVec.resize(topK);
        }
    }

    void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2, NormalizedBBox& intersectBbox) {
        if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin || bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin) {
            intersectBbox.xmin = 0;
            intersectBbox.ymin = 0;
            intersectBbox.xmax = 0;
            intersectBbox.ymax = 0;
        } else {
            intersectBbox.xmin = std::max(bbox1.xmin, bbox2.xmin);
            intersectBbox.ymin = std::max(bbox1.ymin, bbox2.ymin);
            intersectBbox.xmax = std::min(bbox1.xmax, bbox2.xmax);
            intersectBbox.ymax = std::min(bbox1.ymax, bbox2.ymax);
        }
    }

    float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
        NormalizedBBox intersectBbox;
        IntersectBBox(bbox1, bbox2, intersectBbox);
        float intersectWidth, intersectHeight;
        intersectWidth = intersectBbox.xmax - intersectBbox.xmin;
        intersectHeight = intersectBbox.ymax - intersectBbox.ymin;
        if (intersectWidth > 0 && intersectHeight > 0) {
            float intersect_size = intersectWidth * intersectHeight;
            float bbox1_size = BBoxSize(bbox1);
            float bbox2_size = BBoxSize(bbox2);

            return intersect_size / (bbox1_size + bbox2_size - intersect_size);
        } else {
            return 0.0f;
        }
    }

    void caffeNMS(const std::vector<NormalizedBBox>& bboxes, const std::vector<float>& scores,  std::vector<int>& indices) {
        std::vector<std::pair<float, int>> scoreIndexVec;
        GetMaxScoreIndex(scores, attrs.confidence_threshold, attrs.top_k, scoreIndexVec);
        while (scoreIndexVec.size() != 0) {
            const int idx = scoreIndexVec.front().second;
            bool keep = true;
            for (int k = 0; k < indices.size(); ++k) {
                const int kept_idx = indices[k];
                float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
                if (overlap > attrs.nms_threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                indices.push_back(idx);
            }
            scoreIndexVec.erase(scoreIndexVec.begin());
        }
    }

    void mxNetNms(const LabelBBox& decodeBboxesImage, const std::map<int, std::vector<float>>& confScores, std::map<int, std::vector<int>>& indices) {
        std::vector<std::pair<float, std::pair<int, int>>> scoreIndexPairs;
        for (int p = 0; p < numPriors; p++) {
            float conf = -1;
            int id = 0;
            for (int c = 1; c < attrs.num_classes; c++) {
                float temp = confScores.at(c)[p];
                if (temp > conf) {
                    conf = temp;
                    id = c;
                }
            }
            if (id > 0 && conf >= attrs.confidence_threshold) {
                scoreIndexPairs.push_back(std::make_pair(conf, std::make_pair(id, p)));
            }
        }
        std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(), SortScorePairDescend<std::pair<int, int>>);

        if (attrs.top_k != -1)
            if (scoreIndexPairs.size() > attrs.top_k)
                scoreIndexPairs.resize(attrs.top_k);

        while (scoreIndexPairs.size() != 0) {
            const int cls = scoreIndexPairs.front().second.first;
            const int prior = scoreIndexPairs.front().second.second;
            std::vector<int>& currInd = indices[cls];
            bool keep = true;
            for (int i = 0; i < currInd.size(); i++) {
                const int keptIdx = currInd[i];
                auto currBbox = attrs.share_location ? decodeBboxesImage.at(-1) : decodeBboxesImage.at(cls);
                float overlap = JaccardOverlap(currBbox[prior], currBbox[keptIdx]);
                if (overlap > attrs.nms_threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                currInd.push_back(prior);
            }
            scoreIndexPairs.erase(scoreIndexPairs.begin());
        }
    }

  public:
    referenceDetectionOutput(const ngraph::op::DetectionOutputAttrs& _attrs,
                             const std::vector<InferenceEngine::SizeVector>& inputShapes) : attrs(_attrs) {
        numImages = inputShapes[idxLocation][0];
        priorSize = _attrs.normalized ? 4 : 5;
        offset = _attrs.normalized ? 0 : 1;
        numPriors = inputShapes[idxPriors][2] / priorSize;
        numLocClasses = _attrs.share_location ? 1 : static_cast<size_t>(_attrs.num_classes);
    }

    std::vector<float> run(const std::vector<InferenceEngine::Blob::Ptr>& inputs) {
        std::vector<LabelBBox> armLocPreds;
        if (inputs.size() > 4) {
            const float *armLocData = inputs[idxArmLocation]->cbuffer().as<const float *>();
            GetLocPredictions(armLocData, armLocPreds);
        }
        std::vector<LabelBBox> locPreds;
        const float *locData = inputs[idxLocation]->cbuffer().as<const float *>();
        GetLocPredictions(locData, locPreds);
        std::vector<std::map<int, std::vector<float>>> confPreds;
        const float *confData = inputs[idxConfidence]->cbuffer().as<const float *>();
        if (inputs.size() > 3) {
            const float *armConfData = inputs[idxArmConfidence]->cbuffer().as<const float *>();
            OSGetConfidenceScores(confData, armConfData, confPreds);
        } else {
            GetConfidenceScores(confData, confPreds);
        }
        const float *priorData = inputs[idxPriors]->cbuffer().as<const float *>();
        std::vector<std::vector<NormalizedBBox>> priorBboxes;
        std::vector<std::vector<std::vector<float>>> priorVariances;
        GetPriorBBoxes(priorData, priorBboxes, priorVariances);
        std::vector<LabelBBox> decodeBboxes;
        if (inputs.size() > 4) {
            CasRegDecodeBBoxesAll(locPreds, priorBboxes, priorVariances, decodeBboxes, armLocPreds);
        } else {
            DecodeBBoxesAll(locPreds, priorBboxes, priorVariances, decodeBboxes);
        }

        int numKept = 0;
        std::vector<std::map<int, std::vector<int>>> allIndices;
        for (int i = 0; i < numImages; ++i) {
            const LabelBBox& decodeBboxesImage = decodeBboxes[i];
            const std::map<int, std::vector<float>>& confScores = confPreds[i];
            std::map<int, std::vector<int>> indices;
            int numDet = 0;
            if (!attrs.decrease_label_id) {
                // Caffe style
                for (int c = 0; c < attrs.num_classes; ++c) {
                    if (c == attrs.background_label_id) {
                      continue;
                    }
                    const std::vector<float>& scores = confScores.find(c)->second;
                    int label = attrs.share_location ? -1 : c;
                    const std::vector<NormalizedBBox>& bboxes = decodeBboxesImage.find(label)->second;
                    caffeNMS(bboxes, scores, indices[c]);
                    numDet += indices[c].size();
                }
            } else {
                // MXNet style
                mxNetNms(decodeBboxesImage, confScores, indices);
                for (auto it = indices.begin(); it != indices.end(); it++)
                    numDet += it->second.size();
            }
            if (attrs.keep_top_k[0] > -1 && numDet > attrs.keep_top_k[0]) {
                std::vector<std::pair<float, std::pair<int, int>>> scoreIndexPairs;
                for (auto it = indices.begin(); it != indices.end(); ++it) {
                    int label = it->first;
                    const std::vector<int>& labelIndices = it->second;
                    const std::vector<float>& scores = confScores.find(label)->second;
                    for (int j = 0; j < labelIndices.size(); ++j) {
                        int idx = labelIndices[j];
                        scoreIndexPairs.push_back(std::make_pair(scores[idx], std::make_pair(label, idx)));
                    }
                }
                std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(), SortScorePairDescend<std::pair<int, int>>);
                scoreIndexPairs.resize(attrs.keep_top_k[0]);
                std::map<int, std::vector<int>> newIndices;
                for (int j = 0; j < scoreIndexPairs.size(); ++j) {
                    int label = scoreIndexPairs[j].second.first;
                    int idx = scoreIndexPairs[j].second.second;
                    newIndices[label].push_back(idx);
                }
                allIndices.push_back(newIndices);
                numKept += attrs.top_k;
            } else {
                allIndices.push_back(indices);
                numKept += numDet;
            }
        }

        std::vector<float> result(numImages * attrs.keep_top_k[0] * 7, 0);
        int count = 0;
        for (int i = 0; i < numImages; ++i) {
            const std::map<int, std::vector<float>>& confScores = confPreds[i];
            const LabelBBox& decodeBboxesImage = decodeBboxes[i];
            for (auto it = allIndices[i].begin(); it != allIndices[i].end(); ++it) {
                int label = it->first;
                const std::vector<float>& scores = confScores.find(label)->second;
                int loc_label = attrs.share_location ? -1 : label;
                const std::vector<NormalizedBBox>& bboxes = decodeBboxesImage.find(loc_label)->second;
                std::vector<int>& indices = it->second;
                for (int j = 0; j < indices.size(); ++j) {
                    int idx = indices[j];
                    result[count * 7 + 0] = i;
                    result[count * 7 + 1] = attrs.decrease_label_id ? (label - 1) : label;
                    result[count * 7 + 2] = scores[idx];
                    const NormalizedBBox& bbox = bboxes[idx];

                    float xmin = bbox.xmin;
                    float ymin = bbox.ymin;
                    float xmax = bbox.xmax;
                    float ymax = bbox.ymax;

                    if (attrs.clip_after_nms) {
                          xmin = (std::max)(0.0f, (std::min)(1.0f, xmin));
                          ymin = (std::max)(0.0f, (std::min)(1.0f, ymin));
                          xmax = (std::max)(0.0f, (std::min)(1.0f, xmax));
                          ymax = (std::max)(0.0f, (std::min)(1.0f, ymax));
                    }

                    result[count * 7 + 3] = xmin;
                    result[count * 7 + 4] = ymin;
                    result[count * 7 + 5] = xmax;
                    result[count * 7 + 6] = ymax;
                    ++count;
                }
            }
        }
        if (count < numImages * attrs.keep_top_k[0]) {
            result[count * 7 + 0] = -1;
        }
        return result;
    }
};

using DetectionOutputAttributes = std::tuple<
    size_t,             // numClasses
    int,                // topK
    std::vector<int>,   // keepTopK
    std::string,        // codeType
    bool,               // clip_afterNms
    bool,               // clip_beforeNms
    bool                // decreaseLabelId
>;

using ParamsWhichSizeDepends = std::tuple<
    bool,                        // varianceEncodedInTarget
    bool,                        // shareLocation
    bool,                        // normalized
    size_t,                      // inputHeight
    size_t,                      // inputWidth
    InferenceEngine::SizeVector, // "Location" input
    InferenceEngine::SizeVector, // "Confidence" input
    InferenceEngine::SizeVector, // "Priors" input
    InferenceEngine::SizeVector, // "ArmConfidence" input
    InferenceEngine::SizeVector  // "ArmLocation" input
>;

using DetectionOutputParams = std::tuple<
    DetectionOutputAttributes,
    ParamsWhichSizeDepends,
    size_t,     // Number of batch
    std::string // Device name
>;

class DetectionOutputLayerTest : public testing::WithParamInterface<DetectionOutputParams>, public LayerTestsUtils::LayerTestsCommon {
  public:
    static std::string getTestCaseName(testing::TestParamInfo<DetectionOutputParams> obj);
    ngraph::op::DetectionOutputAttrs attrs;
    std::vector<InferenceEngine::SizeVector> inShapes;
    void Infer() override;
    void Validate() override;
  protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
