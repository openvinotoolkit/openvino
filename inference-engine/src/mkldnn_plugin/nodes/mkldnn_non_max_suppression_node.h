// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

class MKLDNNNonMaxSuppressionNode : public MKLDNNNode {
public:
    MKLDNNNonMaxSuppressionNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

    struct filteredBoxes {
        float score;
        int batch_index;
        int class_index;
        int box_index;
        filteredBoxes() = default;
        filteredBoxes(float _score, int _batch_index, int _class_index, int _box_index) :
                score(_score), batch_index(_batch_index), class_index(_class_index), box_index(_box_index) {}
    };

    struct boxInfo {
        float score;
        int idx;
        int suppress_begin_index;
    };

    float intersectionOverUnion(const float *boxesI, const float *boxesJ);

    void nmsWithSoftSigma(const float *boxes, const float *scores, const SizeVector &boxesStrides,
                          const SizeVector &scoresStrides, std::vector<filteredBoxes> &filtBoxes);

    void nmsWithoutSoftSigma(const float *boxes, const float *scores, const SizeVector &boxesStrides,
                             const SizeVector &scoresStrides, std::vector<filteredBoxes> &filtBoxes);

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

    size_t max_output_boxes_per_class = 0lu;
    float iou_threshold = 0.0f;
    float score_threshold = 0.0f;
    float soft_nms_sigma = 0.0f;
    float scale = 1.f;

    SizeVector inputShape_MAXOUTPUTBOXESPERCLASS;
    SizeVector inputShape_IOUTHRESHOLD;
    SizeVector inputShape_SCORETHRESHOLD;
    SizeVector inputShape_SOFTNMSSIGMA;

    SizeVector outputShape_SELECTEDINDICES;
    SizeVector outputShape_SELECTEDSCORES;

    std::string errorPrefix;

    std::vector<std::vector<size_t>> numFiltBox;
    const std::string inType = "input", outType = "output";

    void checkPrecision(const Precision prec, const std::vector<Precision> precList, const std::string name, const std::string type);
    void check1DInput(const SizeVector& dims, const std::vector<Precision> precList, const std::string name, const size_t port);
    void checkOutput(const SizeVector& dims, const std::vector<Precision> precList, const std::string name, const size_t port);
};

}  // namespace MKLDNNPlugin
