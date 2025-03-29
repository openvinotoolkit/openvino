// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <map>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/op/detection_output.hpp"
#include "openvino/op/util/detection_output_base.hpp"

namespace ov {
namespace reference {
enum { idxLocation, idxConfidence, idxPriors, idxArmConfidence, idxArmLocation, numInputs };

template <typename dataType>
class referenceDetectionOutput {
private:
    struct NormalizedBBox {
        dataType xmin = dataType(0);
        dataType ymin = dataType(0);
        dataType xmax = dataType(0);
        dataType ymax = dataType(0);
    };
    using LabelBBox = std::map<int, std::vector<NormalizedBBox>>;

    op::util::DetectionOutputBase::AttributesBase attrs;
    size_t numImages;
    size_t priorSize;
    size_t numPriors;
    size_t priorsBatchSize;
    size_t numLocClasses;
    size_t offset;
    size_t numResults;
    size_t outTotalSize;
    int numClasses;

    void GetLocPredictions(const dataType* locData, std::vector<LabelBBox>& locations) {
        locations.resize(numImages);
        for (size_t i = 0; i < numImages; ++i) {
            LabelBBox& labelBbox = locations[i];
            for (size_t p = 0; p < numPriors; ++p) {
                size_t startIdx = p * numLocClasses * 4;
                for (size_t c = 0; c < numLocClasses; ++c) {
                    int label = attrs.share_location ? -1 : static_cast<int>(c);
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

    void GetConfidenceScores(const dataType* confData, std::vector<std::map<int, std::vector<dataType>>>& confPreds) {
        confPreds.resize(numImages);
        for (size_t i = 0; i < numImages; ++i) {
            std::map<int, std::vector<dataType>>& labelScores = confPreds[i];
            for (size_t p = 0; p < numPriors; ++p) {
                size_t startIdx = p * numClasses;
                for (int c = 0; c < numClasses; ++c) {
                    labelScores[c].push_back(confData[startIdx + c]);
                }
            }
            confData += numPriors * numClasses;
        }
    }

    void OSGetConfidenceScores(const dataType* confData,
                               const dataType* armConfData,
                               std::vector<std::map<int, std::vector<dataType>>>& confPreds) {
        confPreds.resize(numImages);
        for (size_t i = 0; i < numImages; ++i) {
            std::map<int, std::vector<dataType>>& labelScores = confPreds[i];
            for (size_t p = 0; p < numPriors; ++p) {
                size_t startIdx = p * numClasses;
                if (armConfData[p * 2 + 1] < attrs.objectness_score) {
                    for (int c = 0; c < numClasses; ++c) {
                        c == attrs.background_label_id ? labelScores[c].push_back(1) : labelScores[c].push_back(0);
                    }
                } else {
                    for (int c = 0; c < numClasses; ++c) {
                        labelScores[c].push_back(confData[startIdx + c]);
                    }
                }
            }
            confData += numPriors * numClasses;
            armConfData += numPriors * 2;
        }
    }

    dataType BBoxSize(const NormalizedBBox& bbox) {
        if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin) {
            return 0;
        } else {
            dataType width = bbox.xmax - bbox.xmin;
            dataType height = bbox.ymax - bbox.ymin;
            return width * height;
        }
    }

    void GetPriorBBoxes(const dataType* priorData,
                        std::vector<std::vector<NormalizedBBox>>& priorBboxes,
                        std::vector<std::vector<std::vector<dataType>>>& priorVariances) {
        priorBboxes.resize(priorsBatchSize);
        priorVariances.resize(priorsBatchSize);
        int off =
            static_cast<int>(attrs.variance_encoded_in_target ? (numPriors * priorSize) : (2 * numPriors * priorSize));
        for (size_t n = 0; n < priorsBatchSize; n++) {
            std::vector<NormalizedBBox>& currPrBbox = priorBboxes[n];
            std::vector<std::vector<dataType>>& currPrVar = priorVariances[n];
            for (size_t i = 0; i < numPriors; ++i) {
                size_t start_idx = i * priorSize;
                NormalizedBBox bbox;
                bbox.xmin = priorData[start_idx + 0 + offset];
                bbox.ymin = priorData[start_idx + 1 + offset];
                bbox.xmax = priorData[start_idx + 2 + offset];
                bbox.ymax = priorData[start_idx + 3 + offset];
                currPrBbox.push_back(bbox);
            }
            if (!attrs.variance_encoded_in_target) {
                const dataType* priorVar = priorData + numPriors * priorSize;
                for (size_t i = 0; i < numPriors; ++i) {
                    size_t start_idx = i * 4;
                    std::vector<dataType> var(4);
                    for (int j = 0; j < 4; ++j) {
                        var[j] = (priorVar[start_idx + j]);
                    }
                    currPrVar.push_back(var);
                }
            }
            priorData += off;
        }
    }

    void DecodeBBox(const NormalizedBBox& priorBboxes,
                    const std::vector<dataType>& priorVariances,
                    const NormalizedBBox& bbox,
                    NormalizedBBox& decodeBbox) {
        dataType priorXmin = priorBboxes.xmin;
        dataType priorYmin = priorBboxes.ymin;
        dataType priorXmax = priorBboxes.xmax;
        dataType priorYmax = priorBboxes.ymax;

        if (!attrs.normalized) {
            priorXmin /= static_cast<dataType>(attrs.input_width);
            priorYmin /= static_cast<dataType>(attrs.input_height);
            priorXmax /= static_cast<dataType>(attrs.input_width);
            priorYmax /= static_cast<dataType>(attrs.input_height);
        }

        if (attrs.code_type == "caffe.PriorBoxParameter.CORNER") {
            decodeBbox.xmin = priorXmin + priorVariances[0] * bbox.xmin;
            decodeBbox.ymin = priorYmin + priorVariances[1] * bbox.ymin;
            decodeBbox.xmax = priorXmax + priorVariances[2] * bbox.xmax;
            decodeBbox.ymax = priorYmax + priorVariances[3] * bbox.ymax;
        } else if (attrs.code_type == "caffe.PriorBoxParameter.CENTER_SIZE") {
            dataType priorWidth = priorXmax - priorXmin;
            dataType priorHeight = priorYmax - priorYmin;
            dataType priorCenterX = (priorXmin + priorXmax) / 2;
            dataType priorCenterY = (priorYmin + priorYmax) / 2;
            dataType decodeBboxCenterX, decodeBboxCenterY;
            dataType decodeBboxWidth, decodeBboxHeight;
            decodeBboxCenterX = priorVariances[0] * bbox.xmin * priorWidth + priorCenterX;
            decodeBboxCenterY = priorVariances[1] * bbox.ymin * priorHeight + priorCenterY;
            decodeBboxWidth = static_cast<dataType>(std::exp(priorVariances[2] * bbox.xmax)) * priorWidth;
            decodeBboxHeight = static_cast<dataType>(std::exp(priorVariances[3] * bbox.ymax)) * priorHeight;
            decodeBbox.xmin = decodeBboxCenterX - decodeBboxWidth / 2;
            decodeBbox.ymin = decodeBboxCenterY - decodeBboxHeight / 2;
            decodeBbox.xmax = decodeBboxCenterX + decodeBboxWidth / 2;
            decodeBbox.ymax = decodeBboxCenterY + decodeBboxHeight / 2;
        }
    }

    void DecodeBBox(const NormalizedBBox& priorBboxes, const NormalizedBBox& bbox, NormalizedBBox& decodeBbox) {
        dataType priorXmin = priorBboxes.xmin;
        dataType priorYmin = priorBboxes.ymin;
        dataType priorXmax = priorBboxes.xmax;
        dataType priorYmax = priorBboxes.ymax;

        if (!attrs.normalized) {
            priorXmin /= static_cast<dataType>(attrs.input_width);
            priorYmin /= static_cast<dataType>(attrs.input_height);
            priorXmax /= static_cast<dataType>(attrs.input_width);
            priorYmax /= static_cast<dataType>(attrs.input_height);
        }

        if (attrs.code_type == "caffe.PriorBoxParameter.CORNER") {
            decodeBbox.xmin = priorXmin + bbox.xmin;
            decodeBbox.ymin = priorYmin + bbox.ymin;
            decodeBbox.xmax = priorXmax + bbox.xmax;
            decodeBbox.ymax = priorYmax + bbox.ymax;
        } else if (attrs.code_type == "caffe.PriorBoxParameter.CENTER_SIZE") {
            dataType priorWidth = priorXmax - priorXmin;
            dataType priorHeight = priorYmax - priorYmin;
            dataType priorCenterX = (priorXmin + priorXmax) / 2;
            dataType priorCenterY = (priorYmin + priorYmax) / 2;
            dataType decodeBboxCenterX, decodeBboxCenterY;
            dataType decodeBboxWidth, decodeBboxHeight;
            decodeBboxCenterX = bbox.xmin * priorWidth + priorCenterX;
            decodeBboxCenterY = bbox.ymin * priorHeight + priorCenterY;
            decodeBboxWidth = static_cast<dataType>(std::exp(bbox.xmax)) * priorWidth;
            decodeBboxHeight = static_cast<dataType>(std::exp(bbox.ymax)) * priorHeight;
            decodeBbox.xmin = decodeBboxCenterX - decodeBboxWidth / 2;
            decodeBbox.ymin = decodeBboxCenterY - decodeBboxHeight / 2;
            decodeBbox.xmax = decodeBboxCenterX + decodeBboxWidth / 2;
            decodeBbox.ymax = decodeBboxCenterY + decodeBboxHeight / 2;
        }
    }

    void DecodeBBoxes(const std::vector<NormalizedBBox>& priorBboxes,
                      const std::vector<std::vector<dataType>>& priorVariances,
                      const std::vector<NormalizedBBox>& labelLocPreds,
                      std::vector<NormalizedBBox>& decodeBboxes) {
        size_t numBboxes = priorBboxes.size();
        for (size_t i = 0; i < numBboxes; ++i) {
            NormalizedBBox decodeBbox;

            if (attrs.variance_encoded_in_target) {
                DecodeBBox(priorBboxes[i], labelLocPreds[i], decodeBbox);
            } else {
                DecodeBBox(priorBboxes[i], priorVariances[i], labelLocPreds[i], decodeBbox);
            }
            if (attrs.clip_before_nms) {
                decodeBbox.xmin = std::max<dataType>(0, std::min<dataType>(1, decodeBbox.xmin));
                decodeBbox.ymin = std::max<dataType>(0, std::min<dataType>(1, decodeBbox.ymin));
                decodeBbox.xmax = std::max<dataType>(0, std::min<dataType>(1, decodeBbox.xmax));
                decodeBbox.ymax = std::max<dataType>(0, std::min<dataType>(1, decodeBbox.ymax));
            }
            decodeBboxes.push_back(decodeBbox);
        }
    }

    void DecodeBBoxesAll(const std::vector<LabelBBox>& locPreds,
                         const std::vector<std::vector<NormalizedBBox>>& priorBboxes,
                         const std::vector<std::vector<std::vector<dataType>>>& priorVariances,
                         std::vector<LabelBBox>& decodeBboxes) {
        decodeBboxes.resize(numImages);
        for (size_t i = 0; i < numImages; ++i) {
            LabelBBox& decodeBboxesImage = decodeBboxes[i];
            int pboxIdx = static_cast<int>(i);
            if (priorBboxes.size() == 1) {
                pboxIdx = 0;
            }
            const std::vector<NormalizedBBox>& currPrBbox = priorBboxes[pboxIdx];
            const std::vector<std::vector<dataType>>& currPrVar = priorVariances[pboxIdx];
            for (size_t c = 0; c < numLocClasses; ++c) {
                int label = attrs.share_location ? -1 : static_cast<int>(c);
                if (attrs.background_label_id > -1 && label == attrs.background_label_id) {
                    continue;
                }
                const auto& labelLocPreds = locPreds[i].at(label);
                DecodeBBoxes(currPrBbox, currPrVar, labelLocPreds, decodeBboxesImage[label]);
            }
        }
    }

    void CasRegDecodeBBoxesAll(const std::vector<LabelBBox>& locPreds,
                               const std::vector<std::vector<NormalizedBBox>>& priorBboxes,
                               const std::vector<std::vector<std::vector<dataType>>>& priorVariances,
                               std::vector<LabelBBox>& decodeBboxes,
                               const std::vector<LabelBBox>& armLocPreds) {
        decodeBboxes.resize(numImages);
        for (size_t i = 0; i < numImages; ++i) {
            LabelBBox& decodeBboxesImage = decodeBboxes[i];
            const std::vector<NormalizedBBox>& currPrBbox = priorBboxes[i];
            const std::vector<std::vector<dataType>>& currPrVar = priorVariances[i];
            for (size_t c = 0; c < numLocClasses; ++c) {
                int label = attrs.share_location ? -1 : static_cast<int>(c);
                if (attrs.background_label_id > -1 && label == attrs.background_label_id) {
                    continue;
                }
                const auto& labelArmLocPreds = armLocPreds[i].at(label);
                std::vector<NormalizedBBox> decodePriorBboxes;
                DecodeBBoxes(currPrBbox, currPrVar, labelArmLocPreds, decodePriorBboxes);
                const auto& labelLocPreds = locPreds[i].at(label);
                DecodeBBoxes(decodePriorBboxes, currPrVar, labelLocPreds, decodeBboxesImage[label]);
            }
        }
    }

    template <typename T>
    static bool SortScorePairDescend(const std::pair<dataType, T>& pair1, const std::pair<dataType, T>& pair2) {
        return (pair1.first > pair2.first) || (pair1.first == pair2.first && pair1.second < pair2.second);
    }

    void GetMaxScoreIndex(const std::vector<dataType>& scores,
                          const dataType threshold,
                          const int topK,
                          std::vector<std::pair<dataType, int>>& scoreIndexVec) {
        for (size_t i = 0; i < scores.size(); ++i) {
            if (scores[i] > threshold) {
                scoreIndexVec.push_back(std::make_pair(scores[i], static_cast<int>(i)));
            }
        }

        std::stable_sort(scoreIndexVec.begin(), scoreIndexVec.end(), SortScorePairDescend<int>);

        if (topK > -1 && static_cast<size_t>(topK) < scoreIndexVec.size()) {
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
            intersectBbox.xmin = std::max<dataType>(bbox1.xmin, bbox2.xmin);
            intersectBbox.ymin = std::max<dataType>(bbox1.ymin, bbox2.ymin);
            intersectBbox.xmax = std::min<dataType>(bbox1.xmax, bbox2.xmax);
            intersectBbox.ymax = std::min<dataType>(bbox1.ymax, bbox2.ymax);
        }
    }

    dataType JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
        NormalizedBBox intersectBbox;
        IntersectBBox(bbox1, bbox2, intersectBbox);

        dataType intersectWidth, intersectHeight;
        intersectWidth = intersectBbox.xmax - intersectBbox.xmin;
        intersectHeight = intersectBbox.ymax - intersectBbox.ymin;
        if (intersectWidth > 0 && intersectHeight > 0) {
            dataType intersect_size = intersectWidth * intersectHeight;
            dataType bbox1_size = BBoxSize(bbox1);
            dataType bbox2_size = BBoxSize(bbox2);
            return intersect_size / (bbox1_size + bbox2_size - intersect_size);
        } else {
            return static_cast<dataType>(0);
        }
    }

    void caffeNMS(const std::vector<NormalizedBBox>& bboxes,
                  const std::vector<dataType>& scores,
                  std::vector<int>& indices) {
        std::vector<std::pair<dataType, int>> scoreIndexVec;
        GetMaxScoreIndex(scores, static_cast<dataType>(attrs.confidence_threshold), attrs.top_k, scoreIndexVec);
        while (scoreIndexVec.size() != 0) {
            const int idx = scoreIndexVec.front().second;
            bool keep = true;
            for (size_t k = 0; k < indices.size(); ++k) {
                const int kept_idx = indices[k];
                dataType overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);

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

    void mxNetNms(const LabelBBox& decodeBboxesImage,
                  const std::map<int, std::vector<dataType>>& confScores,
                  std::map<int, std::vector<int>>& indices) {
        std::vector<std::pair<dataType, std::pair<int, int>>> scoreIndexPairs;
        for (size_t p = 0; p < numPriors; p++) {
            dataType conf = -1;
            int id = 0;
            for (int c = 1; c < numClasses; c++) {
                if (attrs.background_label_id > -1 && c == attrs.background_label_id)
                    continue;
                dataType temp = confScores.at(c)[p];
                if (temp > conf) {
                    conf = temp;
                    id = c;
                }
            }
            if (id > 0 && conf >= attrs.confidence_threshold) {
                scoreIndexPairs.push_back(std::make_pair(conf, std::make_pair(id, static_cast<int>(p))));
            }
        }
        std::sort(
            scoreIndexPairs.begin(),
            scoreIndexPairs.end(),
            [](const std::pair<dataType, std::pair<int, int>>& p1, const std::pair<dataType, std::pair<int, int>>& p2) {
                return (p1.first > p2.first) || (p1.first == p2.first && p1.second.second < p2.second.second);
            });

        if (attrs.top_k != -1)
            if (scoreIndexPairs.size() > static_cast<size_t>(attrs.top_k))
                scoreIndexPairs.resize(attrs.top_k);

        while (scoreIndexPairs.size() != 0) {
            const int cls = scoreIndexPairs.front().second.first;
            const int prior = scoreIndexPairs.front().second.second;
            std::vector<int>& currInd = indices[cls];
            bool keep = true;
            for (size_t i = 0; i < currInd.size(); i++) {
                const int keptIdx = currInd[i];
                auto currBbox = attrs.share_location ? decodeBboxesImage.at(-1) : decodeBboxesImage.at(cls);
                dataType overlap = JaccardOverlap(currBbox[prior], currBbox[keptIdx]);
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
    referenceDetectionOutput(const op::v0::DetectionOutput::Attributes& _attrs,
                             const Shape& locShape,
                             const Shape& priorsShape,
                             const Shape& outShape)
        : attrs(_attrs) {
        numImages = locShape[0];
        priorSize = _attrs.normalized ? 4 : 5;
        offset = _attrs.normalized ? 0 : 1;
        numPriors = priorsShape[2] / priorSize;
        priorsBatchSize = priorsShape[0];
        numClasses = _attrs.num_classes;
        numLocClasses = _attrs.share_location ? 1 : numClasses;
        numResults = outShape[2];
        outTotalSize = shape_size(outShape);
    }

    referenceDetectionOutput(const op::util::DetectionOutputBase::AttributesBase& _attrs,
                             const Shape& locShape,
                             const Shape& classPredShape,
                             const Shape& priorsShape,
                             const Shape& outShape)
        : attrs(_attrs) {
        numImages = locShape[0];
        priorSize = _attrs.normalized ? 4 : 5;
        offset = _attrs.normalized ? 0 : 1;
        numPriors = priorsShape[2] / priorSize;
        priorsBatchSize = priorsShape[0];
        numClasses = classPredShape[1] / static_cast<int>(numPriors);
        numLocClasses = _attrs.share_location ? 1 : numClasses;
        numResults = outShape[2];
        outTotalSize = shape_size(outShape);
    }

    void run(const dataType* _location,
             const dataType* _confidence,
             const dataType* _priors,
             const dataType* _armConfidence,
             const dataType* _armLocation,
             dataType* result) {
        std::fill(result, result + outTotalSize, dataType{0});
        bool withAddBoxPred = _armConfidence != nullptr && _armLocation != nullptr;
        std::vector<LabelBBox> armLocPreds;
        if (withAddBoxPred) {
            GetLocPredictions(_armLocation, armLocPreds);
        }
        std::vector<LabelBBox> locPreds;
        GetLocPredictions(_location, locPreds);
        std::vector<std::map<int, std::vector<dataType>>> confPreds;
        if (withAddBoxPred) {
            OSGetConfidenceScores(_confidence, _armConfidence, confPreds);
        } else {
            GetConfidenceScores(_confidence, confPreds);
        }
        std::vector<std::vector<NormalizedBBox>> priorBboxes;
        std::vector<std::vector<std::vector<dataType>>> priorVariances;
        GetPriorBBoxes(_priors, priorBboxes, priorVariances);
        std::vector<LabelBBox> decodeBboxes;
        if (withAddBoxPred) {
            CasRegDecodeBBoxesAll(locPreds, priorBboxes, priorVariances, decodeBboxes, armLocPreds);
        } else {
            DecodeBBoxesAll(locPreds, priorBboxes, priorVariances, decodeBboxes);
        }

        std::vector<std::map<int, std::vector<int>>> allIndices;
        for (size_t i = 0; i < numImages; ++i) {
            const LabelBBox& decodeBboxesImage = decodeBboxes[i];
            const std::map<int, std::vector<dataType>>& confScores = confPreds[i];
            std::map<int, std::vector<int>> indices;
            int numDet = 0;
            if (!attrs.decrease_label_id) {
                // Caffe style
                for (int c = 0; c < numClasses; ++c) {
                    if (c == attrs.background_label_id) {
                        continue;
                    }
                    const auto conf_score = confScores.find(c);
                    if (conf_score == confScores.end())
                        continue;
                    const std::vector<dataType>& scores = conf_score->second;

                    int label = attrs.share_location ? -1 : c;
                    const auto decode_bboxes = decodeBboxesImage.find(label);
                    if (decode_bboxes == decodeBboxesImage.end())
                        continue;
                    const std::vector<NormalizedBBox>& bboxes = decode_bboxes->second;
                    caffeNMS(bboxes, scores, indices[c]);
                    numDet += static_cast<int>(indices[c].size());
                }
            } else {
                // MXNet style
                mxNetNms(decodeBboxesImage, confScores, indices);
                for (auto it = indices.begin(); it != indices.end(); it++)
                    numDet += static_cast<int>(it->second.size());
            }
            if (attrs.keep_top_k[0] > -1 && numDet > attrs.keep_top_k[0]) {
                std::vector<std::pair<dataType, std::pair<int, int>>> scoreIndexPairs;
                for (auto it = indices.begin(); it != indices.end(); ++it) {
                    int label = it->first;
                    const std::vector<int>& labelIndices = it->second;
                    const auto conf_score = confScores.find(label);
                    if (conf_score == confScores.end())
                        continue;
                    const std::vector<dataType>& scores = conf_score->second;
                    for (size_t j = 0; j < labelIndices.size(); ++j) {
                        int idx = labelIndices[j];
                        scoreIndexPairs.push_back(std::make_pair(scores[idx], std::make_pair(label, idx)));
                    }
                }
                std::sort(scoreIndexPairs.begin(),
                          scoreIndexPairs.end(),
                          [](const std::pair<dataType, std::pair<int, int>>& p1,
                             const std::pair<dataType, std::pair<int, int>>& p2) {
                              return (p1.first > p2.first) ||
                                     (p1.first == p2.first && p1.second.second < p2.second.second);
                          });
                scoreIndexPairs.resize(attrs.keep_top_k[0]);
                std::map<int, std::vector<int>> newIndices;
                for (size_t j = 0; j < scoreIndexPairs.size(); ++j) {
                    int label = scoreIndexPairs[j].second.first;
                    int idx = scoreIndexPairs[j].second.second;
                    newIndices[label].push_back(idx);
                }
                allIndices.push_back(newIndices);
            } else {
                allIndices.push_back(indices);
            }
        }

        size_t count = 0;
        for (size_t i = 0; i < numImages; ++i) {
            const std::map<int, std::vector<dataType>>& confScores = confPreds[i];
            const LabelBBox& decodeBboxesImage = decodeBboxes[i];
            for (auto it = allIndices[i].begin(); it != allIndices[i].end(); ++it) {
                int label = it->first;
                const std::vector<dataType>& scores = confScores.at(label);
                int loc_label = attrs.share_location ? -1 : label;
                const auto decode_bboxes = decodeBboxesImage.find(loc_label);
                if (decode_bboxes == decodeBboxesImage.end())
                    continue;
                const std::vector<NormalizedBBox>& bboxes = decode_bboxes->second;
                std::vector<int>& indices = it->second;
                for (size_t j = 0; j < indices.size(); ++j) {
                    int idx = indices[j];
                    result[count * 7 + 0] = static_cast<dataType>(i);
                    result[count * 7 + 1] = static_cast<dataType>(attrs.decrease_label_id ? (label - 1) : label);
                    result[count * 7 + 2] = scores[idx];
                    const NormalizedBBox& bbox = bboxes[idx];

                    dataType xmin = bbox.xmin;
                    dataType ymin = bbox.ymin;
                    dataType xmax = bbox.xmax;
                    dataType ymax = bbox.ymax;

                    if (attrs.clip_after_nms) {
                        xmin = std::max<dataType>(0, std::min<dataType>(1, xmin));
                        ymin = std::max<dataType>(0, std::min<dataType>(1, ymin));
                        xmax = std::max<dataType>(0, std::min<dataType>(1, xmax));
                        ymax = std::max<dataType>(0, std::min<dataType>(1, ymax));
                    }

                    result[count * 7 + 3] = xmin;
                    result[count * 7 + 4] = ymin;
                    result[count * 7 + 5] = xmax;
                    result[count * 7 + 6] = ymax;
                    ++count;
                }
            }
        }
        if (count < numResults) {
            result[count * 7 + 0] = -1;
        }
    }
};
}  // namespace reference
}  // namespace ov
