// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#if defined(HAVE_AVX2)
#include <immintrin.h>
#endif
#include "ie_parallel.hpp"


namespace {
struct Indexer {
  const std::vector<int> dims_;
  int total_{1};

  explicit Indexer(const std::vector<int>& dims) : dims_(dims) {
      total_ = 1;
      for (size_t i = 0; i < dims_.size(); ++i) {
          total_ *= dims_[i];
      }
  }

  int operator()(const std::vector<int>& idx) const {
      int flat_idx = 0;
      assert(idx.size() == dims_.size());
      for (size_t i = 0; i < dims_.size(); ++i) {
          assert(0 <= idx[i] && idx[i] < dims_[i]);
          flat_idx = flat_idx * dims_[i] + idx[i];
      }
      assert(flat_idx < total_);
      return flat_idx;
  }
};
}  // namespace


namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

static
void refine_anchors(const float* deltas, const float* scores, const float* anchors,
                    float* proposals, const int anchors_num, const int bottom_H,
                    const int bottom_W, const float img_H, const float img_W,
                    const float min_box_H, const float min_box_W,
                    const float max_delta_log_wh,
                    float coordinates_offset) {
    Indexer delta_idx({anchors_num, 4, bottom_H, bottom_W});
    Indexer score_idx({anchors_num, 1, bottom_H, bottom_W});
    Indexer proposal_idx({bottom_H, bottom_W, anchors_num, 5});
    Indexer anchor_idx({bottom_H, bottom_W, anchors_num, 4});

    parallel_for2d(bottom_H, bottom_W, [&](int h, int w) {
            for (int anchor = 0; anchor < anchors_num; ++anchor) {
                float x0 = anchors[anchor_idx({h, w, anchor, 0})];
                float y0 = anchors[anchor_idx({h, w, anchor, 1})];
                float x1 = anchors[anchor_idx({h, w, anchor, 2})];
                float y1 = anchors[anchor_idx({h, w, anchor, 3})];

                const float dx = deltas[delta_idx({anchor, 0, h, w})];
                const float dy = deltas[delta_idx({anchor, 1, h, w})];
                const float d_log_w = deltas[delta_idx({anchor, 2, h, w})];
                const float d_log_h = deltas[delta_idx({anchor, 3, h, w})];

                const float score = scores[score_idx({anchor, 0, h, w})];

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

                proposals[proposal_idx({h, w, anchor, 0})] = x0;
                proposals[proposal_idx({h, w, anchor, 1})] = y0;
                proposals[proposal_idx({h, w, anchor, 2})] = x1;
                proposals[proposal_idx({h, w, anchor, 3})] = y1;
                proposals[proposal_idx({h, w, anchor, 4})] = (min_box_W <= box_w) * (min_box_H <= box_h) * score;
            }
    });
}

static void unpack_boxes(const float* p_proposals, float* unpacked_boxes, int pre_nms_topn) {
    parallel_for(pre_nms_topn, [&](size_t i) {
        unpacked_boxes[0*pre_nms_topn + i] = p_proposals[5*i + 0];
        unpacked_boxes[1*pre_nms_topn + i] = p_proposals[5*i + 1];
        unpacked_boxes[2*pre_nms_topn + i] = p_proposals[5*i + 2];
        unpacked_boxes[3*pre_nms_topn + i] = p_proposals[5*i + 3];
        unpacked_boxes[4*pre_nms_topn + i] = p_proposals[5*i + 4];
    });
}

static
void nms_cpu(const int num_boxes, int is_dead[],
             const float* boxes, int index_out[], int* const num_out,
             const int base_index, const float nms_thresh, const int max_num_out,
             float coordinates_offset) {
    const int num_proposals = num_boxes;
    int count = 0;

    const float* x0 = boxes + 0 * num_proposals;
    const float* y0 = boxes + 1 * num_proposals;
    const float* x1 = boxes + 2 * num_proposals;
    const float* y1 = boxes + 3 * num_proposals;

    memset(is_dead, 0, num_boxes * sizeof(int));

#if defined(HAVE_AVX2)
    __m256  vc_fone = _mm256_set1_ps(coordinates_offset);
    __m256i vc_ione = _mm256_set1_epi32(1);
    __m256  vc_zero = _mm256_set1_ps(0.0f);

    __m256 vc_nms_thresh = _mm256_set1_ps(nms_thresh);
#endif

    for (int box = 0; box < num_boxes; ++box) {
        if (is_dead[box])
            continue;

        index_out[count++] = base_index + box;
        if (count == max_num_out)
            break;

        int tail = box + 1;

#if defined(HAVE_AVX2)
        __m256 vx0i = _mm256_set1_ps(x0[box]);
        __m256 vy0i = _mm256_set1_ps(y0[box]);
        __m256 vx1i = _mm256_set1_ps(x1[box]);
        __m256 vy1i = _mm256_set1_ps(y1[box]);

        __m256 vA_width  = _mm256_sub_ps(vx1i, vx0i);
        __m256 vA_height = _mm256_sub_ps(vy1i, vy0i);
        __m256 vA_area   = _mm256_mul_ps(_mm256_add_ps(vA_width, vc_fone), _mm256_add_ps(vA_height, vc_fone));

        for (; tail <= num_boxes - 8; tail += 8) {
            __m256i *pdst = reinterpret_cast<__m256i*>(is_dead + tail);
            __m256i  vdst = _mm256_loadu_si256(pdst);

            __m256 vx0j = _mm256_loadu_ps(x0 + tail);
            __m256 vy0j = _mm256_loadu_ps(y0 + tail);
            __m256 vx1j = _mm256_loadu_ps(x1 + tail);
            __m256 vy1j = _mm256_loadu_ps(y1 + tail);

            __m256 vx0 = _mm256_max_ps(vx0i, vx0j);
            __m256 vy0 = _mm256_max_ps(vy0i, vy0j);
            __m256 vx1 = _mm256_min_ps(vx1i, vx1j);
            __m256 vy1 = _mm256_min_ps(vy1i, vy1j);

            __m256 vwidth  = _mm256_add_ps(_mm256_sub_ps(vx1, vx0), vc_fone);
            __m256 vheight = _mm256_add_ps(_mm256_sub_ps(vy1, vy0), vc_fone);
            __m256 varea = _mm256_mul_ps(_mm256_max_ps(vc_zero, vwidth), _mm256_max_ps(vc_zero, vheight));

            __m256 vB_width  = _mm256_sub_ps(vx1j, vx0j);
            __m256 vB_height = _mm256_sub_ps(vy1j, vy0j);
            __m256 vB_area   = _mm256_mul_ps(_mm256_add_ps(vB_width, vc_fone), _mm256_add_ps(vB_height, vc_fone));

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
                const float width  = std::max<float>(0.0f,  x1 - x0 + coordinates_offset);
                const float height = std::max<float>(0.0f,  y1 - y0 + coordinates_offset);
                const float area   = width * height;

                // area of A, B
                const float A_area = (x1i - x0i + coordinates_offset) * (y1i - y0i + coordinates_offset);
                const float B_area = (x1j - x0j + coordinates_offset) * (y1j - y0j + coordinates_offset);

                // IoU
                res = area / (A_area + B_area - area);
            }

            if (nms_thresh < res)
                is_dead[tail] = 1;
        }
    }

    *num_out = count;
}


static
void fill_output_blobs(const float* proposals, const int* roi_indices,
                       float* rois, float* scores,
                       const int num_proposals, const int num_rois, const int post_nms_topn) {
    const float *src_x0 = proposals + 0 * num_proposals;
    const float *src_y0 = proposals + 1 * num_proposals;
    const float *src_x1 = proposals + 2 * num_proposals;
    const float *src_y1 = proposals + 3 * num_proposals;
    const float *src_score = proposals + 4 * num_proposals;

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


class ONNXCustomProposalImpl : public ExtLayerBase {
private:
    const int INPUT_IM_INFO {0};
    const int INPUT_ANCHORS {1};
    const int INPUT_DELTAS {2};
    const int INPUT_SCORES {3};
    const int OUTPUT_ROIS {0};
    const int OUTPUT_SCORES {1};

public:
    explicit ONNXCustomProposalImpl(const CNNLayer *layer) {
        try {
            if (layer->insData.size() != 4 || layer->outData.size() != 2)
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            min_size_ = layer->GetParamAsFloat("min_size");
            nms_thresh_ = layer->GetParamAsFloat("nms_threshold");
            pre_nms_topn_ = layer->GetParamAsInt("pre_nms_count");
            post_nms_topn_ = layer->GetParamAsInt("post_nms_count");

            coordinates_offset = 0.0f;

            roi_indices_.resize(post_nms_topn_);
            addConfig(layer,
                      {DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN),
                       DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)},
                      {DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    void print_shape(const Blob::Ptr& b) {
        for (size_t i = 0; i < b->getTensorDesc().getDims().size(); ++i) {
            std::cout << b->getTensorDesc().getDims()[i] << ", ";
        }
        std::cout << std::endl;
    }

    StatusCode execute(std::vector<Blob::Ptr> &inputs, std::vector<Blob::Ptr> &outputs,
                       ResponseDesc *resp) noexcept override {
        if (inputs.size() != 4 || outputs.size() != 2) {
            if (resp) {
                std::string errorMsg = "Incorrect number of input or output edges!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }

        // Prepare memory
        const float* p_deltas_item = inputs[INPUT_DELTAS]->buffer();
        const float* p_scores_item = inputs[INPUT_SCORES]->buffer();
        const float* p_anchors_item = inputs[INPUT_ANCHORS]->buffer();
        const float* p_img_info_cpu = inputs[INPUT_IM_INFO]->buffer();

        float* p_roi_item = outputs[OUTPUT_ROIS]->buffer();
        float* p_roi_score_item = outputs[OUTPUT_SCORES]->buffer();


        size_t img_info_size = 1;
        for (size_t i = 0; i < inputs[INPUT_IM_INFO]->getTensorDesc().getDims().size(); i++) {
            img_info_size *= inputs[INPUT_IM_INFO]->getTensorDesc().getDims()[i];
        }

        const int anchors_num = inputs[INPUT_SCORES]->getTensorDesc().getDims()[0];

        // bottom shape: (num_anchors) x H x W
        const int bottom_H = inputs[INPUT_DELTAS]->getTensorDesc().getDims()[1];
        const int bottom_W = inputs[INPUT_DELTAS]->getTensorDesc().getDims()[2];

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
            refine_anchors(p_deltas_item, p_scores_item, p_anchors_item,
                           reinterpret_cast<float *>(&proposals_[0]), anchors_num, bottom_H,
                           bottom_W, img_H, img_W,
                           min_box_H, min_box_W,
                           static_cast<const float>(log(1000. / 16.)),
                           1.0f);
            std::partial_sort(proposals_.begin(), proposals_.begin() + pre_nms_topn, proposals_.end(),
                              [](const ProposalBox& struct1, const ProposalBox& struct2) {
                                  return (struct1.score > struct2.score);
                              });

            unpack_boxes(reinterpret_cast<float *>(&proposals_[0]), &unpacked_boxes[0], pre_nms_topn);
            nms_cpu(pre_nms_topn, &is_dead[0], &unpacked_boxes[0], &roi_indices_[0], &num_rois, 0,
                    nms_thresh_, post_nms_topn_, coordinates_offset);
            fill_output_blobs(&unpacked_boxes[0], &roi_indices_[0], p_roi_item, p_roi_score_item,
                              pre_nms_topn, num_rois, post_nms_topn_);
        }

        return OK;
    }

private:
    float min_size_;
    int pre_nms_topn_;
    int post_nms_topn_;
    float nms_thresh_;
    float coordinates_offset;

    std::vector<int> roi_indices_;
};

REG_FACTORY_FOR(ImplFactory<ONNXCustomProposalImpl>, ExperimentalDetectronGenerateProposalsSingleImage);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
