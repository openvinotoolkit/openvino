//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Elementwise addition operation.
            ///
            class NGRAPH_API NonMaxSuppression : public Op
            {
            public:
                enum class BoxEncodingType
                {
                    CORNER,
                    CENTER
                };

                static constexpr NodeTypeInfo type_info{"NonMaxSuppression", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                NonMaxSuppression() = default;

                /// \brief Constructs a NonMaxSuppression operation.
                ///
                /// \param boxes Node producing the box coordinates
                /// \param scores Node producing the box scores
                /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
                /// selected per class
                /// \param iou_threshold Node producing intersection over union threshold
                /// \param score_threshold Node producing minimum score threshold
                /// \param box_encoding Specifies the format of boxes data encoding
                NonMaxSuppression(const Output<Node>& boxes,
                                  const Output<Node>& scores,
                                  const Output<Node>& max_output_boxes_per_class,
                                  const Output<Node>& iou_threshold,
                                  const Output<Node>& score_threshold,
                                  const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                                  const bool sort_result_descending = true);

                /// \brief Constructs a NonMaxSuppression operation with default values for the last
                ///        3 inputs
                ///
                /// \param boxes Node producing the box coordinates
                /// \param scores Node producing the box coordinates
                /// \param box_encoding Specifies the format of boxes data encoding
                /// \param sort_result_descending Specifies whether it is necessary to sort selected
                /// boxes across batches
                /// \param output_type Specifies the output tensor type
                NonMaxSuppression(const Output<Node>& boxes,
                                  const Output<Node>& scores,
                                  const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                                  const bool sort_result_descending = true);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                BoxEncodingType get_box_encoding() const { return m_box_encoding; }
                void set_box_encoding(const BoxEncodingType box_encoding)
                {
                    m_box_encoding = box_encoding;
                }
                bool get_sort_result_descending() const { return m_sort_result_descending; }
                void set_sort_result_descending(const bool sort_result_descending)
                {
                    m_sort_result_descending = sort_result_descending;
                }

            protected:
                BoxEncodingType m_box_encoding = BoxEncodingType::CORNER;
                bool m_sort_result_descending = true;

            private:
                int64_t max_boxes_output_from_input() const;
            };
        } // namespace v1

        namespace v3
        {
            /// \brief NonMaxSuppression operation
            ///
            class NGRAPH_API NonMaxSuppression : public Op
            {
            public:
                enum class BoxEncodingType
                {
                    CORNER,
                    CENTER
                };

                static constexpr NodeTypeInfo type_info{"NonMaxSuppression", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                NonMaxSuppression() = default;

                /// \brief Constructs a NonMaxSuppression operation.
                ///
                /// \param boxes Node producing the box coordinates
                /// \param scores Node producing the box scores
                /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
                /// selected per class
                /// \param iou_threshold Node producing intersection over union threshold
                /// \param score_threshold Node producing minimum score threshold
                /// \param box_encoding Specifies the format of boxes data encoding
                /// \param sort_result_descending Specifies whether it is necessary to sort selected
                /// boxes across batches
                /// \param output_type Specifies the output tensor type
                NonMaxSuppression(const Output<Node>& boxes,
                                  const Output<Node>& scores,
                                  const Output<Node>& max_output_boxes_per_class,
                                  const Output<Node>& iou_threshold,
                                  const Output<Node>& score_threshold,
                                  const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                                  const bool sort_result_descending = true,
                                  const ngraph::element::Type& output_type = ngraph::element::i64);

                /// \brief Constructs a NonMaxSuppression operation with default values for the last
                ///        3 inputs
                ///
                /// \param boxes Node producing the box coordinates
                /// \param scores Node producing the box coordinates
                /// \param box_encoding Specifies the format of boxes data encoding
                /// \param sort_result_descending Specifies whether it is necessary to sort selected
                /// boxes across batches
                /// \param output_type Specifies the output tensor type
                NonMaxSuppression(const Output<Node>& boxes,
                                  const Output<Node>& scores,
                                  const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                                  const bool sort_result_descending = true,
                                  const ngraph::element::Type& output_type = ngraph::element::i64);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                BoxEncodingType get_box_encoding() const { return m_box_encoding; }
                void set_box_encoding(const BoxEncodingType box_encoding)
                {
                    m_box_encoding = box_encoding;
                }
                bool get_sort_result_descending() const { return m_sort_result_descending; }
                void set_sort_result_descending(const bool sort_result_descending)
                {
                    m_sort_result_descending = sort_result_descending;
                }

                element::Type get_output_type() const { return m_output_type; }
                void set_output_type(const element::Type& output_type)
                {
                    m_output_type = output_type;
                }
                using Node::set_output_type;

            protected:
                BoxEncodingType m_box_encoding = BoxEncodingType::CORNER;
                bool m_sort_result_descending = true;
                ngraph::element::Type m_output_type = ngraph::element::i64;

            private:
                int64_t max_boxes_output_from_input() const;
            };
        } // namespace v3

        namespace v4
        {
            /// \brief NonMaxSuppression operation
            ///
            class NGRAPH_API NonMaxSuppression : public op::v3::NonMaxSuppression
            {
            public:
                static constexpr NodeTypeInfo type_info{"NonMaxSuppression", 4};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                NonMaxSuppression() = default;

                /// \brief Constructs a NonMaxSuppression operation.
                ///
                /// \param boxes Node producing the box coordinates
                /// \param scores Node producing the box scores
                /// \param max_output_boxes_per_class Node producing maximum number of boxes to be
                /// selected per class
                /// \param iou_threshold Node producing intersection over union threshold
                /// \param score_threshold Node producing minimum score threshold
                /// \param box_encoding Specifies the format of boxes data encoding
                /// \param sort_result_descending Specifies whether it is necessary to sort selected
                /// boxes across batches
                /// \param output_type Specifies the output tensor type
                NonMaxSuppression(const Output<Node>& boxes,
                                  const Output<Node>& scores,
                                  const Output<Node>& max_output_boxes_per_class,
                                  const Output<Node>& iou_threshold,
                                  const Output<Node>& score_threshold,
                                  const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                                  const bool sort_result_descending = true,
                                  const ngraph::element::Type& output_type = ngraph::element::i64);

                /// \brief Constructs a NonMaxSuppression operation with default values for the last
                ///        3 inputs
                ///
                /// \param boxes Node producing the box coordinates
                /// \param scores Node producing the box coordinates
                /// \param box_encoding Specifies the format of boxes data encoding
                /// \param sort_result_descending Specifies whether it is necessary to sort selected
                /// boxes across batches
                /// \param output_type Specifies the output tensor type
                NonMaxSuppression(const Output<Node>& boxes,
                                  const Output<Node>& scores,
                                  const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                                  const bool sort_result_descending = true,
                                  const ngraph::element::Type& output_type = ngraph::element::i64);

                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v4
    }     // namespace op

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s,
                             const op::v1::NonMaxSuppression::BoxEncodingType& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v1::NonMaxSuppression::BoxEncodingType>
        : public EnumAttributeAdapterBase<op::v1::NonMaxSuppression::BoxEncodingType>
    {
    public:
        AttributeAdapter(op::v1::NonMaxSuppression::BoxEncodingType& value)
            : EnumAttributeAdapterBase<op::v1::NonMaxSuppression::BoxEncodingType>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v1::NonMaxSuppression::BoxEncodingType>", 1};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s,
                             const op::v3::NonMaxSuppression::BoxEncodingType& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v3::NonMaxSuppression::BoxEncodingType>
        : public EnumAttributeAdapterBase<op::v3::NonMaxSuppression::BoxEncodingType>
    {
    public:
        AttributeAdapter(op::v3::NonMaxSuppression::BoxEncodingType& value)
            : EnumAttributeAdapterBase<op::v3::NonMaxSuppression::BoxEncodingType>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v3::NonMaxSuppression::BoxEncodingType>", 1};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
} // namespace ngraph
