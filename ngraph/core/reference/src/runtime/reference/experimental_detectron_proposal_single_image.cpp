// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/experimental_detectron_proposal_single_image.hpp"
#include <algorithm>
#include <cassert>
#include <utility>
#include "ngraph/op/experimental_detectron_generate_proposals.hpp"
#include "ngraph/shape.hpp"

namespace
{
} // namespace

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void experimental_detectron_proposals_single_image(
                const float* im_info,
                const float* anchors,
                const float* deltas,
                const float* scores,
                const op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes& attrs,
                float* output_rois,
                float* output_scores)
            {
            }

            void experimental_detectron_proposals_single_image_postprocessing(
                void* prois,
                void* pscores,
                const ngraph::element::Type output_type,
                const std::vector<float>& output_rois,
                const std::vector<float>& output_scores,
                const Shape& output_rois_shape,
                const Shape& output_scores_shape)
            {
                size_t rois_num = output_rois_shape[0];

                switch (output_type)
                {
                case element::Type_t::bf16:
                {
                    bfloat16* rois_ptr = reinterpret_cast<bfloat16*>(prois);
                    bfloat16* scores_ptr = reinterpret_cast<bfloat16*>(pscores);
                    for (size_t i = 0; i < rois_num; ++i)
                    {
                        rois_ptr[4 * i + 0] = bfloat16(output_rois[4 * i + 0]);
                        rois_ptr[4 * i + 1] = bfloat16(output_rois[4 * i + 1]);
                        rois_ptr[4 * i + 2] = bfloat16(output_rois[4 * i + 2]);
                        rois_ptr[4 * i + 3] = bfloat16(output_rois[4 * i + 3]);
                        scores_ptr[i] = bfloat16(output_scores[i]);
                    }
                }
                break;
                case element::Type_t::f16:
                {
                    float16* rois_ptr = reinterpret_cast<float16*>(prois);
                    float16* scores_ptr = reinterpret_cast<float16*>(pscores);
                    for (size_t i = 0; i < rois_num; ++i)
                    {
                        rois_ptr[4 * i + 0] = float16(output_rois[4 * i + 0]);
                        rois_ptr[4 * i + 1] = float16(output_rois[4 * i + 1]);
                        rois_ptr[4 * i + 2] = float16(output_rois[4 * i + 2]);
                        rois_ptr[4 * i + 3] = float16(output_rois[4 * i + 3]);
                        scores_ptr[i] = float16(output_scores[i]);
                    }
                }
                break;
                case element::Type_t::f32:
                {
                    float* rois_ptr = reinterpret_cast<float*>(prois);
                    float* scores_ptr = reinterpret_cast<float*>(pscores);
                    memcpy(rois_ptr,
                           output_rois.data(),
                           shape_size(output_rois_shape) * sizeof(float));
                    memcpy(scores_ptr,
                           output_scores.data(),
                           shape_size(output_scores_shape) * sizeof(float));
                }
                break;
                default:;
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
