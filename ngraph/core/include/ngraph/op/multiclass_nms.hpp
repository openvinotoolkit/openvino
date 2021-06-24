// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/nms_base.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v8
        {
            /// \brief MulticlassNms operation
            ///
            class NGRAPH_API MulticlassNms : public util::NmsBase
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                MulticlassNms() = default;

                /// \brief Constructs a MulticlassNms operation
                ///
                /// \param boxes Node producing the box coordinates
                /// \param scores Node producing the box scores
                /// \param sort_result Specifies order of output elements
                /// \param sort_result_across_batch Specifies  whenever it is necessary to
                /// sort selected boxes across batches or not
                /// \param output_type Specifies the output tensor type
                /// \param iou_threshold Specifies intersection over union threshold
                /// \param score_threshold Specifies minimum score to consider box for the
                /// processing
                /// \param nms_top_k Specifies maximum number of boxes to be selected per
                /// class, -1 meaning to keep all boxes
                /// \param keep_top_k Specifies maximum number of boxes to be selected per
                /// batch element, -1 meaning to keep all boxes
                /// \param background_class Specifies the background class id, -1 meaning to keep
                /// all classes
                /// \param nms_eta Specifies eta parameter for adpative NMS, in close range [0, 1.0]
                /// \param normalized Specifies whether boxes are normalized or not
                MulticlassNms(const Output<Node>& boxes,
                              const Output<Node>& scores,
                              const SortResultType sort_result_type = SortResultType::NONE,
                              bool sort_result_across_batch = false,
                              const ngraph::element::Type& output_type = ngraph::element::i64,
                              const float iou_threshold = 0.0f,
                              const float score_threshold = 0.0f,
                              const int nms_top_k = -1,
                              const int keep_top_k = -1,
                              const int background_class = -1,
                              const float nms_eta = 1.0f,
                              const bool normalized = true);

                bool visit_attributes(AttributeVisitor& visitor) override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool get_sort_result_across_batch() const { return m_sort_result_across_batch; }
                void set_sort_result_across_batch(const bool sort_result_across_batch)
                {
                    m_sort_result_across_batch = sort_result_across_batch;
                }

                float get_iou_threshold() const { return m_iou_threshold; }
                void set_iou_threshold(const float iou_threshold)
                {
                    m_iou_threshold = iou_threshold;
                }

                float get_score_threshold() const { return m_score_threshold; }
                void set_score_threshold(const float score_threshold)
                {
                    m_score_threshold = score_threshold;
                }

                int get_background_class() const { return m_background_class; }
                void set_background_class(const int background_class)
                {
                    m_background_class = background_class;
                }

                float get_nms_eta() const { return m_nms_eta; }
                void set_nms_eta(const float nms_eta) { m_nms_eta = nms_eta; }

                bool get_normalized() const { return m_normalized; }
                void set_normalized(const bool normalized) { m_normalized = normalized; }

            protected:
                bool m_sort_result_across_batch;
                float m_iou_threshold;
                float m_score_threshold;
                int m_background_class;
                float m_nms_eta;
                bool m_normalized;
                void validate() override;
            };
        } // namespace v8
    }     // namespace op
} // namespace ngraph
