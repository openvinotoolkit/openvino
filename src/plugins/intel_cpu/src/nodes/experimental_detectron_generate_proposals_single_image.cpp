// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(HAVE_AVX2)
#    include <immintrin.h>
#endif

#include "common/cpu_memcpy.h"
#include "experimental_detectron_generate_proposals_single_image.h"
#include "openvino/core/parallel.hpp"
#include "openvino/op/experimental_detectron_generate_proposals.hpp"

namespace ov::intel_cpu::node {
namespace {

struct Indexer4d {
    int dim3_;
    int dim23_;
    int dim123_;

    explicit Indexer4d(int dim0, int dim1, int dim2, int dim3)
        : dim3_(dim3),
          dim23_(dim2 * dim3),
          dim123_(dim1 * dim2 * dim3) {
        (void)dim0;
    }

    int operator()(int i, int j, int k, int n) const {
        return i * dim123_ + j * dim23_ + k * dim3_ + n;
    }
};

void refine_anchors(const float* deltas,
                    const float* scores,
                    const float* anchors,
                    float* proposals,
                    const int anchors_num,
                    const int bottom_H,
                    const int bottom_W,
                    const float img_H,
                    const float img_W,
                    const float min_box_H,
                    const float min_box_W,
                    const float max_delta_log_wh,
                    float coordinates_offset) {
    Indexer4d delta_idx(anchors_num, 4, bottom_H, bottom_W);
    Indexer4d score_idx(anchors_num, 1, bottom_H, bottom_W);
    Indexer4d proposal_idx(bottom_H, bottom_W, anchors_num, 5);
    Indexer4d anchor_idx(bottom_H, bottom_W, anchors_num, 4);

    parallel_for2d(bottom_H, bottom_W, [&](int h, int w) {
        for (int anchor = 0; anchor < anchors_num; ++anchor) {
            int a_idx = anchor_idx(h, w, anchor, 0);
            float x0 = anchors[a_idx + 0];
            float y0 = anchors[a_idx + 1];
            float x1 = anchors[a_idx + 2];
            float y1 = anchors[a_idx + 3];

            const float dx = deltas[delta_idx(anchor, 0, h, w)];
            const float dy = deltas[delta_idx(anchor, 1, h, w)];
            const float d_log_w = deltas[delta_idx(anchor, 2, h, w)];
            const float d_log_h = deltas[delta_idx(anchor, 3, h, w)];

            const float score = scores[score_idx(anchor, 0, h, w)];

            // width & height of box
            const float ww = x1 - x0 + coordinates_offset;
            const float hh = y1 - y0 + coordinates_offset;
            // center location of box
            const float ctr_x = x0 + 0.5f * ww;
            const float ctr_y = y0 + 0.5f * hh;

            // new center location according to deltas (dx, dy)
            const float pred_ctr_x = dx * ww + ctr_x;
            const float pred_ctr_y = dy * hh + ctr_y;
            // new width & height according to deltas d(log w), d(log h)
            const float pred_w = std::exp(std::min(d_log_w, max_delta_log_wh)) * ww;
            const float pred_h = std::exp(std::min(d_log_h, max_delta_log_wh)) * hh;

            // update upper-left corner location
            x0 = pred_ctr_x - 0.5f * pred_w;
            y0 = pred_ctr_y - 0.5f * pred_h;
            // update lower-right corner location
            x1 = pred_ctr_x + 0.5f * pred_w - coordinates_offset;
            y1 = pred_ctr_y + 0.5f * pred_h - coordinates_offset;

            // adjust new corner locations to be within the image region,
            x0 = std::max<float>(0.0f, std::min<float>(x0, img_W - coordinates_offset));
            y0 = std::max<float>(0.0f, std::min<float>(y0, img_H - coordinates_offset));
            x1 = std::max<float>(0.0f, std::min<float>(x1, img_W - coordinates_offset));
            y1 = std::max<float>(0.0f, std::min<float>(y1, img_H - coordinates_offset));

            // recompute new width & height
            const float box_w = x1 - x0 + coordinates_offset;
            const float box_h = y1 - y0 + coordinates_offset;

            int p_idx = proposal_idx(h, w, anchor, 0);
            proposals[p_idx + 0] = x0;
            proposals[p_idx + 1] = y0;
            proposals[p_idx + 2] = x1;
            proposals[p_idx + 3] = y1;
            proposals[p_idx + 4] = (min_box_W <= box_w) * (min_box_H <= box_h) * score;
        }
    });
}

void unpack_boxes(const float* p_proposals, float* unpacked_boxes, int pre_nms_topn) {
    parallel_for(pre_nms_topn, [&](size_t i) {
        unpacked_boxes[0 * pre_nms_topn + i] = p_proposals[5 * i + 0];
        unpacked_boxes[1 * pre_nms_topn + i] = p_proposals[5 * i + 1];
        unpacked_boxes[2 * pre_nms_topn + i] = p_proposals[5 * i + 2];
        unpacked_boxes[3 * pre_nms_topn + i] = p_proposals[5 * i + 3];
        unpacked_boxes[4 * pre_nms_topn + i] = p_proposals[5 * i + 4];
    });
}

void nms_cpu(const int num_boxes,
             int is_dead[],
             const float* boxes,
             int index_out[],
             int* const num_out,
             const int base_index,
             const float nms_thresh,
             const int max_num_out,
             float coordinates_offset) {
    const int num_proposals = num_boxes;
    int count = 0;

    const float* x0 = boxes + 0 * num_proposals;
    const float* y0 = boxes + 1 * num_proposals;
    const float* x1 = boxes + 2 * num_proposals;
    const float* y1 = boxes + 3 * num_proposals;

    std::memset(is_dead, 0, num_boxes * sizeof(int));

#if defined(HAVE_AVX2)
    __m256 vc_fone = _mm256_set1_ps(coordinates_offset);
    __m256i vc_ione = _mm256_set1_epi32(1);
    __m256 vc_zero = _mm256_set1_ps(0.0f);

    __m256 vc_nms_thresh = _mm256_set1_ps(nms_thresh);
#endif

    for (int box = 0; box < num_boxes; ++box) {
        if (is_dead[box]) {
            continue;
        }

        index_out[count++] = base_index + box;
        if (count == max_num_out) {
            break;
        }

        int tail = box + 1;

#if defined(HAVE_AVX2)
        __m256 vx0i = _mm256_set1_ps(x0[box]);
        __m256 vy0i = _mm256_set1_ps(y0[box]);
        __m256 vx1i = _mm256_set1_ps(x1[box]);
        __m256 vy1i = _mm256_set1_ps(y1[box]);

        __m256 vA_width = _mm256_sub_ps(vx1i, vx0i);
        __m256 vA_height = _mm256_sub_ps(vy1i, vy0i);
        __m256 vA_area = _mm256_mul_ps(_mm256_add_ps(vA_width, vc_fone), _mm256_add_ps(vA_height, vc_fone));

        for (; tail <= num_boxes - 8; tail += 8) {
            __m256i* pdst = reinterpret_cast<__m256i*>(is_dead + tail);
            __m256i vdst = _mm256_loadu_si256(pdst);

            __m256 vx0j = _mm256_loadu_ps(x0 + tail);
            __m256 vy0j = _mm256_loadu_ps(y0 + tail);
            __m256 vx1j = _mm256_loadu_ps(x1 + tail);
            __m256 vy1j = _mm256_loadu_ps(y1 + tail);

            __m256 vx0 = _mm256_max_ps(vx0i, vx0j);
            __m256 vy0 = _mm256_max_ps(vy0i, vy0j);
            __m256 vx1 = _mm256_min_ps(vx1i, vx1j);
            __m256 vy1 = _mm256_min_ps(vy1i, vy1j);

            __m256 vwidth = _mm256_add_ps(_mm256_sub_ps(vx1, vx0), vc_fone);
            __m256 vheight = _mm256_add_ps(_mm256_sub_ps(vy1, vy0), vc_fone);
            __m256 varea = _mm256_mul_ps(_mm256_max_ps(vc_zero, vwidth), _mm256_max_ps(vc_zero, vheight));

            __m256 vB_width = _mm256_sub_ps(vx1j, vx0j);
            __m256 vB_height = _mm256_sub_ps(vy1j, vy0j);
            __m256 vB_area = _mm256_mul_ps(_mm256_add_ps(vB_width, vc_fone), _mm256_add_ps(vB_height, vc_fone));

            __m256 vdivisor = _mm256_sub_ps(_mm256_add_ps(vA_area, vB_area), varea);
            __m256 vintersection_area = _mm256_div_ps(varea, vdivisor);

            __m256 vcmp_0 = _mm256_cmp_ps(vx0i, vx1j, _CMP_LE_OS);
            __m256 vcmp_1 = _mm256_cmp_ps(vy0i, vy1j, _CMP_LE_OS);
            __m256 vcmp_2 = _mm256_cmp_ps(vx0j, vx1i, _CMP_LE_OS);
            __m256 vcmp_3 = _mm256_cmp_ps(vy0j, vy1i, _CMP_LE_OS);
            __m256 vcmp_4 = _mm256_cmp_ps(vc_nms_thresh, vintersection_area, _CMP_LT_OS);

            vcmp_0 = _mm256_and_ps(vcmp_0, vcmp_1);
            vcmp_2 = _mm256_and_ps(vcmp_2, vcmp_3);
            vcmp_4 = _mm256_and_ps(vcmp_4, vcmp_0);
            vcmp_4 = _mm256_and_ps(vcmp_4, vcmp_2);

            _mm256_storeu_si256(pdst, _mm256_blendv_epi8(vdst, vc_ione, _mm256_castps_si256(vcmp_4)));
        }
#endif

        for (; tail < num_boxes; ++tail) {
            float res = 0.0f;

            const float x0i = x0[box];
            const float y0i = y0[box];
            const float x1i = x1[box];
            const float y1i = y1[box];

            const float x0j = x0[tail];
            const float y0j = y0[tail];
            const float x1j = x1[tail];
            const float y1j = y1[tail];

            if (x0i <= x1j && y0i <= y1j && x0j <= x1i && y0j <= y1i) {
                // overlapped region (= box)
                const float x0 = std::max<float>(x0i, x0j);
                const float y0 = std::max<float>(y0i, y0j);
                const float x1 = std::min<float>(x1i, x1j);
                const float y1 = std::min<float>(y1i, y1j);

                // intersection area
                const float width = std::max<float>(0.0f, x1 - x0 + coordinates_offset);
                const float height = std::max<float>(0.0f, y1 - y0 + coordinates_offset);
                const float area = width * height;

                // area of A, B
                const float A_area = (x1i - x0i + coordinates_offset) * (y1i - y0i + coordinates_offset);
                const float B_area = (x1j - x0j + coordinates_offset) * (y1j - y0j + coordinates_offset);

                // IoU
                res = area / (A_area + B_area - area);
            }

            if (nms_thresh < res) {
                is_dead[tail] = 1;
            }
        }
    }

    *num_out = count;
}

void fill_output_blobs(const float* proposals,
                       const int* roi_indices,
                       float* rois,
                       float* scores,
                       const int num_proposals,
                       const int num_rois,
                       const int post_nms_topn) {
    const float* src_x0 = proposals + 0 * num_proposals;
    const float* src_y0 = proposals + 1 * num_proposals;
    const float* src_x1 = proposals + 2 * num_proposals;
    const float* src_y1 = proposals + 3 * num_proposals;
    const float* src_score = proposals + 4 * num_proposals;

    parallel_for(num_rois, [&](size_t i) {
        int index = roi_indices[i];
        rois[i * 4 + 0] = src_x0[index];
        rois[i * 4 + 1] = src_y0[index];
        rois[i * 4 + 2] = src_x1[index];
        rois[i * 4 + 3] = src_y1[index];
        scores[i] = src_score[index];
    });

    if (num_rois < post_nms_topn) {
        for (int i = 4 * num_rois; i < 4 * post_nms_topn; i++) {
            rois[i] = 0.f;
        }
        for (int i = num_rois; i < post_nms_topn; i++) {
            scores[i] = 0.f;
        }
    }
}

}  // namespace

bool ExperimentalDetectronGenerateProposalsSingleImage::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                                                             std::string& errorMessage) noexcept {
    try {
        const auto proposalOp =
            ov::as_type_ptr<const ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(op);
        if (!proposalOp) {
            errorMessage = "Node is not an instance of the Proposal from the operations set v0.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ExperimentalDetectronGenerateProposalsSingleImage::ExperimentalDetectronGenerateProposalsSingleImage(
    const std::shared_ptr<ov::Node>& op,
    const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    auto proposalOp = ov::as_type_ptr<const ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(op);
    auto proposalAttrs = proposalOp->get_attrs();

    min_size_ = proposalAttrs.min_size;
    nms_thresh_ = proposalAttrs.nms_threshold;
    pre_nms_topn_ = proposalAttrs.pre_nms_count;
    post_nms_topn_ = proposalAttrs.post_nms_count;

    coordinates_offset = 0.0f;

    roi_indices_.resize(post_nms_topn_);
}

void ExperimentalDetectronGenerateProposalsSingleImage::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32},
                          {LayoutType::ncsp, ov::element::f32},
                          {LayoutType::ncsp, ov::element::f32},
                          {LayoutType::ncsp, ov::element::f32}},
                         {{LayoutType::ncsp, ov::element::f32}, {LayoutType::ncsp, ov::element::f32}},
                         impl_desc_type::ref_any);
}

void ExperimentalDetectronGenerateProposalsSingleImage::execute(const dnnl::stream& strm) {
    try {
        if (inputShapes.size() != 4 || outputShapes.size() != 2) {
            THROW_CPU_NODE_ERR("Incorrect number of input or output edges!");
        }

        size_t anchor_dims_size = 1;
        const auto& anchorDims = getParentEdgeAt(INPUT_ANCHORS)->getMemory().getStaticDims();
        for (uint64_t anchorDim : anchorDims) {
            anchor_dims_size *= anchorDim;
        }

        size_t deltas_dims_size = 1;
        const auto& deltaDims = getParentEdgeAt(INPUT_DELTAS)->getMemory().getStaticDims();
        for (uint64_t deltaDim : deltaDims) {
            deltas_dims_size *= deltaDim;
        }
        if (anchor_dims_size != deltas_dims_size) {
            THROW_CPU_NODE_ERR("'Anchors' blob size for ONNXProposal is incompatible with 'deltas' blob size!");
        }

        size_t score_dims_size = 1;
        const auto& scoreDims = getParentEdgeAt(INPUT_SCORES)->getMemory().getStaticDims();
        for (uint64_t scoreDim : scoreDims) {
            score_dims_size *= scoreDim;
        }
        if (deltas_dims_size != (4 * score_dims_size)) {
            THROW_CPU_NODE_ERR("'Deltas' blob size for ONNXProposal is incompatible with 'scores' blob size!");
        }

        // Prepare memory
        const auto* p_deltas_item = getSrcDataAtPortAs<const float>(INPUT_DELTAS);
        const auto* p_scores_item = getSrcDataAtPortAs<const float>(INPUT_SCORES);
        const auto* p_anchors_item = getSrcDataAtPortAs<const float>(INPUT_ANCHORS);
        const auto* p_img_info_cpu = getSrcDataAtPortAs<const float>(INPUT_IM_INFO);

        auto* p_roi_item = getDstDataAtPortAs<float>(OUTPUT_ROIS);
        auto* p_roi_score_item = getDstDataAtPortAs<float>(OUTPUT_SCORES);

        const int anchors_num = scoreDims[0];

        // bottom shape: (num_anchors) x H x W
        const int bottom_H = deltaDims[1];
        const int bottom_W = deltaDims[2];

        // input image height & width
        const float img_H = p_img_info_cpu[0];
        const float img_W = p_img_info_cpu[1];

        // scale factor for height & width

        // minimum box width & height
        const float min_box_H = min_size_;
        const float min_box_W = min_size_;

        // number of all proposals = num_anchors * H * W
        const int num_proposals = anchors_num * bottom_H * bottom_W;

        // number of top-n proposals before NMS
        const int pre_nms_topn = std::min<int>(num_proposals, pre_nms_topn_);

        // number of final RoIs
        int num_rois = 0;

        // enumerate all proposals
        //   num_proposals = num_anchors * H * W
        //   (x1, y1, x2, y2, score) for each proposal
        // NOTE: for bottom, only foreground scores are passed
        struct ProposalBox {
            float x0;
            float y0;
            float x1;
            float y1;
            float score;
        };
        std::vector<ProposalBox> proposals_(num_proposals);
        std::vector<float> unpacked_boxes(5 * pre_nms_topn);
        std::vector<int> is_dead(pre_nms_topn);

        // Execute
        int batch_size = 1;  // inputs[INPUT_DELTAS]->getTensorDesc().getDims()[0];
        for (int n = 0; n < batch_size; ++n) {
            refine_anchors(p_deltas_item,
                           p_scores_item,
                           p_anchors_item,
                           reinterpret_cast<float*>(&proposals_[0]),
                           anchors_num,
                           bottom_H,
                           bottom_W,
                           img_H,
                           img_W,
                           min_box_H,
                           min_box_W,
                           static_cast<const float>(std::log(1000. / 16.)),
                           1.0f);
            std::partial_sort(proposals_.begin(),
                              proposals_.begin() + pre_nms_topn,
                              proposals_.end(),
                              [](const ProposalBox& struct1, const ProposalBox& struct2) {
                                  return (struct1.score > struct2.score);
                              });

            unpack_boxes(reinterpret_cast<float*>(&proposals_[0]), &unpacked_boxes[0], pre_nms_topn);
            nms_cpu(pre_nms_topn,
                    &is_dead[0],
                    &unpacked_boxes[0],
                    &roi_indices_[0],
                    &num_rois,
                    0,
                    nms_thresh_,
                    post_nms_topn_,
                    coordinates_offset);
            fill_output_blobs(&unpacked_boxes[0],
                              &roi_indices_[0],
                              p_roi_item,
                              p_roi_score_item,
                              pre_nms_topn,
                              num_rois,
                              post_nms_topn_);
        }
    } catch (const std::exception& e) {
        THROW_CPU_NODE_ERR(e.what());
    }
}

bool ExperimentalDetectronGenerateProposalsSingleImage::created() const {
    return getType() == Type::ExperimentalDetectronGenerateProposalsSingleImage;
}

bool ExperimentalDetectronGenerateProposalsSingleImage::needShapeInfer() const {
    return false;
}

bool ExperimentalDetectronGenerateProposalsSingleImage::needPrepareParams() const {
    return false;
}

}  // namespace ov::intel_cpu::node
