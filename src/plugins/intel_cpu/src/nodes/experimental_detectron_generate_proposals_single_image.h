// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class ExperimentalDetectronGenerateProposalsSingleImage : public Node {
public:
    ExperimentalDetectronGenerateProposalsSingleImage(const std::shared_ptr<ov::Node>& op,
                                                      const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    // Inputs:
    //      rois, shape [n, 4]
    //      rois_probs, shape [n]
    // Outputs:
    //      top_rois, shape [max_rois, 4]

    const int INPUT_IM_INFO{0};
    const int INPUT_ANCHORS{1};
    const int INPUT_DELTAS{2};
    const int INPUT_SCORES{3};
    const int OUTPUT_ROIS{0};
    const int OUTPUT_SCORES{1};

    float min_size_;
    int pre_nms_topn_;
    int post_nms_topn_;
    float nms_thresh_;
    float coordinates_offset;

    std::vector<int> roi_indices_;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
