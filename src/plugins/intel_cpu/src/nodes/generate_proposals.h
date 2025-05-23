// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class GenerateProposals : public Node {
public:
    GenerateProposals(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    // Inputs:
    //      im_info, shape [N, 3] or [N, 4]
    //      anchors, shape [H, W, A, 4]
    //      deltas,  shape [N, A * 4, H, W]
    //      scores,  shape [N, A, H, W]
    // Outputs:
    //      rois,    shape [rois_num, 4]
    //      scores,  shape [rois_num]
    //      roi_num, shape [N]

    const int INPUT_IM_INFO{0};
    const int INPUT_ANCHORS{1};
    const int INPUT_DELTAS{2};
    const int INPUT_SCORES{3};
    const int OUTPUT_ROIS{0};
    const int OUTPUT_SCORES{1};
    const int OUTPUT_ROI_NUM{2};

    float min_size_ = 0.f;
    int pre_nms_topn_ = 0;
    int post_nms_topn_ = 0;
    float nms_thresh_ = 0.f;
    float coordinates_offset_ = 0.f;

    std::vector<int> roi_indices_;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
