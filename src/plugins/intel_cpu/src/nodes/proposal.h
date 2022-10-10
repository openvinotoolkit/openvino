// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
 \
#pragma once

#include <ie_common.h>
#include <node.h>
#include "kernels/jit_uni_nms_proposal_kernel.hpp"
#include <memory>

namespace ov {
namespace intel_cpu {
namespace node {

struct proposal_conf {
    size_t feat_stride_;
    size_t base_size_;
    size_t min_size_;
    int pre_nms_topn_;
    int post_nms_topn_;
    float nms_thresh_;
    float box_coordinate_scale_;
    float box_size_scale_;
    std::vector<float> scales;
    std::vector<float> ratios;
    bool normalize_;

    size_t anchors_shape_0;

    // Framework specific parameters
    float coordinates_offset;
    bool swap_xy;
    bool initial_clip;     // clip initial bounding boxes
    bool clip_before_nms;  // clip bounding boxes before nms step
    bool clip_after_nms;   // clip bounding boxes after nms step
    bool round_ratios;     // round ratios during anchors generation stage
    bool shift_anchors;    // shift anchors by half size of the box
};

class Proposal : public Node {
public:
    Proposal(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool needPrepareParams() const override { return false; };
    void createPrimitive() override;
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    void executeImpl(const float *input0, const float *input1, std::vector<size_t> dims0, std::array<float, 4> img_info,
                     const float *anchors, int *roi_indices, float *output0, float *output1, proposal_conf &conf);

private:
    const size_t PROBABILITIES_IN_IDX = 0lu;
    const size_t ANCHORS_IN_IDX = 1lu;
    const size_t IMG_INFO_IN_IDX = 2lu;
    const size_t ROI_OUT_IDX = 0lu;
    const size_t PROBABILITIES_OUT_IDX = 1lu;

    proposal_conf conf;
    std::vector<float> anchors;
    std::vector<int> roi_indices;
    bool store_prob;  // store blob with proposal probabilities
    std::unique_ptr<jit_uni_nms_proposal_kernel> nms_kernel_ {};
};

void nms_cpu(const int num_boxes, int is_dead[], const float *boxes, int index_out[], std::size_t *const num_out,
             const int base_index, const float nms_thresh, const int max_num_out, float coordinates_offset);

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
