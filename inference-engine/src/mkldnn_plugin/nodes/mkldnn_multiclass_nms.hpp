// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <memory>
#include <mkldnn_node.h>
#include <string>
#include <vector>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

class MKLDNNMultiClassNmsNode : public MKLDNNNode {
public:
  MKLDNNMultiClassNmsNode(const std::shared_ptr<ngraph::Node> &op,
                                        const mkldnn::engine &eng,
                                        MKLDNNWeightsSharing::Ptr &cache);

  void getSupportedDescriptors() override{};
  void initSupportedPrimitiveDescriptors() override;
  void createPrimitive() override{};
  void execute(mkldnn::stream strm) override;
  bool created() const override;

  static bool isSupportedOperation(const std::shared_ptr<ngraph::Node> &op,
                                   std::string &errorMessage) noexcept;

  struct filteredBoxes {
    float score;
    int batch_index;
    int class_index;
    int box_index;
    filteredBoxes() = default;
    filteredBoxes(float _score, int _batch_index, int _class_index,
                  int _box_index)
        : score(_score), batch_index(_batch_index), class_index(_class_index),
          box_index(_box_index) {}
  };

  struct boxInfo {
    float score;
    int idx;
    int suppress_begin_index;
  };

  float intersectionOverUnion(const float *boxesI, const float *boxesJ,
                              const bool normalized);

  void nmsWithEta(const float *boxes, const float *scores,
                  const SizeVector &boxesStrides,
                  const SizeVector &scoresStrides,
                  std::vector<filteredBoxes> &filtBoxes);

  void nmsWithoutEta(const float *boxes, const float *scores,
                     const SizeVector &boxesStrides,
                     const SizeVector &scoresStrides,
                     std::vector<filteredBoxes> &filtBoxes);

private:
  // input (port Num)
  const size_t NMS_BOXES = 0;
  const size_t NMS_SCORES = 1;

  // output (port Num)
  const size_t NMS_SELECTEDOUTPUTS = 0;
  const size_t NMS_SELECTEDINDICES = 1;
  const size_t NMS_SELECTEDNUM = 2;

  bool sort_result_across_batch = false;
  int32_t sort_result_type = 2;

  size_t num_batches;
  size_t num_boxes;
  size_t num_classes;

  int max_output_boxes_per_class = 0;
  float iou_threshold = 0.0f;
  float score_threshold = 0.0f;

  int32_t background_class = 0;
  int32_t keep_top_k = 0;
  float nms_eta = 0.0f;
  bool normalized = true;

  SizeVector outputShape_SELECTEDINDICES;
  SizeVector outputShape_SELECTEDOUTPUTS;
  std::string errorPrefix;

  std::vector<std::vector<size_t>> numFiltBox;
  std::vector<size_t> numBoxOffset;
  const std::string inType = "input", outType = "output";

  void checkPrecision(const Precision prec,
                      const std::vector<Precision> precList,
                      const std::string name, const std::string type);
  void checkOutput(const SizeVector &dims,
                   const std::vector<Precision> precList,
                   const std::string name, const size_t port);
};

} // namespace MKLDNNPlugin
