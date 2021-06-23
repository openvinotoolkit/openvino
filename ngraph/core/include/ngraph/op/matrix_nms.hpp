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
            /// \brief MatrixNms operation
            ///
            class NGRAPH_API MatrixNms : public util::NmsBase
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                enum class DecayFunction
                {
                    GAUSSIAN,
                    LINEAR
                };

                MatrixNms() = default;

                /// \brief Constructs a MatrixNms operation
                ///
                /// \param boxes Node producing the box coordinates
                /// \param scores Node producing the box scores
                /// \param sort_result Specifies order of output elements
                /// \param sort_result_across_batch Specifies whenever it is necessary to
                /// sort selected boxes across batches or not
                /// \param output_type Specifies the output tensor type
                /// \param score_threshold Specifies minimum score to consider box for the
                /// processing
                /// \param nms_top_k Specifies maximum number of boxes to be selected per
                /// class, -1 meaning to keep all boxes
                /// \param keep_top_k Specifies maximum number of boxes to be selected per
                /// batch element, -1 meaning to keep all boxes
                /// \param background_class Specifies the background class id, -1 meaning to keep
                /// all classes
                /// \param decay_function Specifies decay function used to decay scores
                /// \param gaussian_sigma Specifies gaussian_sigma parameter for gaussian
                /// decay_function \param post_threshold Specifies threshold to filter out boxes
                /// with low confidence score after decaying
                /// \param normalized Specifies whether boxes are normalized or not
                MatrixNms(const Output<Node>& boxes,
                          const Output<Node>& scores,
                          const SortResultType sort_result_type = SortResultType::NONE,
                          const bool sort_result_across_batch = false,
                          const ngraph::element::Type& output_type = ngraph::element::i64,
                          const float score_threshold = 0.0f,
                          const int nms_top_k = -1,
                          const int keep_top_k = -1,
                          const int background_class = -1,
                          const DecayFunction decay_function = DecayFunction::LINEAR,
                          const float gaussian_sigma = 2.0f,
                          const float post_threshold = 0.0f,
                          const bool normalized = true);

                bool visit_attributes(AttributeVisitor& visitor) override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool get_sort_result_across_batch() const { return m_sort_result_across_batch; }
                void set_sort_result_across_batch(const bool sort_result_across_batch)
                {
                    m_sort_result_across_batch = sort_result_across_batch;
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

                DecayFunction get_decay_function() const { return m_decay_function; }
                void set_decay_function(const DecayFunction decay_function)
                {
                    m_decay_function = decay_function;
                }

                float get_gaussian_sigma() const { return m_gaussian_sigma; }
                void set_gaussian_sigma(const float gaussian_sigma)
                {
                    m_gaussian_sigma = gaussian_sigma;
                }

                float get_post_threshold() const { return m_post_threshold; }
                void set_post_threshold(const float post_threshold)
                {
                    m_post_threshold = post_threshold;
                }

                bool get_normalized() const { return m_normalized; }
                void set_normalized(const bool normalized) { m_normalized = normalized; }

            protected:
                bool m_sort_result_across_batch;
                float m_score_threshold;
                int m_background_class;
                DecayFunction m_decay_function;
                float m_gaussian_sigma;
                float m_post_threshold;
                bool m_normalized;
                void validate() override;
            };
        } // namespace v8
    }     // namespace op
    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const op::v8::MatrixNms::DecayFunction& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v8::MatrixNms::DecayFunction>
        : public EnumAttributeAdapterBase<op::v8::MatrixNms::DecayFunction>
    {
    public:
        AttributeAdapter(op::v8::MatrixNms::DecayFunction& value)
            : EnumAttributeAdapterBase<op::v8::MatrixNms::DecayFunction>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v8::MatrixNms::DecayFunction>", 1};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
} // namespace ngraph
