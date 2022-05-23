// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>

namespace ov {
namespace intel_cpu {
namespace node {

class GenerateProposals : public Node {
public:
    GenerateProposals(const std::shared_ptr<ngraph::Node>& op,
        const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override;
    void executeDynamicImpl(dnnl::stream strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

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

    const int INPUT_IM_INFO {0};
    const int INPUT_ANCHORS {1};
    const int INPUT_DELTAS {2};
    const int INPUT_SCORES {3};
    const int OUTPUT_ROIS {0};
    const int OUTPUT_SCORES {1};
    const int OUTPUT_ROI_NUM {2};

    float min_size_;
    int pre_nms_topn_;
    int post_nms_topn_;
    float nms_thresh_;
    float coordinates_offset_;

    std::vector<int> roi_indices_;
};

}   // namespace node
}  // namespace intel_cpu
}  // namespace ov
