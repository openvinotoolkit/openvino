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

#include <memory>

#include "ngraph/axis_set.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            // \brief Computes indices of top k maximum/minimum index along a specified axis for a
            //        given tensor
            class NGRAPH_API TopK : public Op
            {
            public:
                using SortType = TopKSortType;

                static constexpr NodeTypeInfo type_info{"TopK", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a TopK operation
                TopK() = default;
                /// \brief Constructs a TopK operation.
                ///
                /// \param arg The input tensor
                /// \param top_k_axis The axis along which to compute top k indices
                /// \param index_element_type produce indices. Currently, only int64 or int32 are
                ///                           supported
                /// \param k Number of top indices to compute. Compute all indices if k = 0
                /// \param compute_max Compute top k max or top k min?
                /// \param sort SortType for sorting results, default - SORT_VALUES
                TopK(const Output<Node>& arg,
                     size_t top_k_axis,
                     const element::Type& index_element_type,
                     size_t k = 0,
                     bool compute_max = true,
                     SortType sort = SortType::SORT_VALUES);
                /// \brief Constructs a TopK operation.
                ///
                /// \param arg The input tensor
                /// \param k Number of top indices to compute. Compute all indices if k = 0
                /// \param top_k_axis The axis along which to compute top k indices
                /// \param index_element_type produce indices. Currently, only int64 or int32 are
                ///                           supported
                /// \param compute_max Compute top k max or top k min?
                /// \param sort SortType for sorting results, default - SORT_VALUES
                TopK(const Output<Node>& arg,
                     const Output<Node>& k,
                     size_t top_k_axis,
                     const element::Type& index_element_type,
                     bool compute_max = true,
                     SortType sort = SortType::SORT_VALUES);

                /// \brief Constructs a TopK operation.
                ///
                /// \param arg The input tensor
                /// \param k Number of top indices to compute. Compute all indices if k = 0
                /// \param top_k_axis The axis along which to compute top k indices
                /// \param index_element_type produce indices. Currently, only int64 or int32 are
                /// supported
                /// \param compute_max Compute top k max or top k min?
                /// \param sort SortType for sorting results, default - NONE
                TopK(const Output<Node>& arg,
                     const Output<Node>& k,
                     const Output<Node>& top_k_axis,
                     const element::Type& index_element_type,
                     bool compute_max = true,
                     SortType sort = SortType::NONE);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                size_t get_k() const;
                void set_k(size_t k);

                size_t get_top_k_axis() const;
                Dimension get_top_k_axis_dynamic() const;
                void set_top_k_axis(size_t k);

                element::Type get_index_element_type() const { return m_index_element_type; }
                bool get_compute_max() const { return m_compute_max; }
                SortType get_sort() const { return m_sort; }
                size_t get_default_output_index() const override { return no_default_index(); }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;

            protected:
                element::Type m_index_element_type;
                bool m_compute_max{false};
                SortType m_sort{SortType::NONE};
                Shape compute_output_shape(const Shape input_shape,
                                           const int64_t k,
                                           const size_t axis);
            };
        } // namespace v0

        namespace v1
        {
            /// \brief Computes indices and values of the k maximum/minimum values
            ///        for each slice along specified axis.
            class NGRAPH_API TopK : public Op
            {
            public:
                using SortType = TopKSortType;
                using Mode = TopKMode;

                static constexpr NodeTypeInfo type_info{"TopK", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a TopK operation
                TopK() = default;
                /// \brief Constructs a TopK operation with two outputs: values and indices.
                ///        By default the indices output is described by i32 data type.
                ///
                /// \param data The input tensor
                /// \param k Specifies how many maximum/minimum elements should be computed
                ///          (note: scalar input tensor)
                /// \param axis The axis along which to compute top k indices
                /// \param mode Specifies which operation (min or max) is used to select
                ///             the biggest element of two.
                /// \param sort Specifies order of output elements and/or indices
                ///             Accepted values: none, index, value
                /// \param index_element_type Specyfies type of produced indices
                TopK(const Output<Node>& data,
                     const Output<Node>& k,
                     const int64_t axis,
                     const std::string& mode,
                     const std::string& sort,
                     const element::Type& index_element_type = element::i32);

                TopK(const Output<Node>& data,
                     const Output<Node>& k,
                     const int64_t axis,
                     const Mode mode,
                     const SortType sort,
                     const element::Type& index_element_type = element::i32);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual size_t get_version() const override { return 1; }
                /// \brief Returns axis value after normalization
                /// \note If input rank required to normalization is dynamic, the exception is
                /// thrown
                uint64_t get_axis() const;
                /// \brief Returns axis value before normalization
                int64_t get_provided_axis() const { return m_axis; }
                void set_axis(const int64_t axis);
                Mode get_mode() const { return m_mode; }
                void set_mode(const Mode mode) { m_mode = mode; }
                SortType get_sort_type() const { return m_sort; }
                void set_sort_type(const SortType sort) { m_sort = sort; }
                element::Type get_index_element_type() const { return m_index_element_type; }
                void set_index_element_type(const element::Type& index_element_type)
                {
                    m_index_element_type = index_element_type;
                }
                /// \brief Returns the value of K, if available
                ///
                /// \note If the second input to this op is a constant, the value is retrieved
                ///       and returned. If the input is not constant(dynamic) this method returns 0
                size_t get_k() const;
                void set_k(size_t k);
                size_t get_default_output_index() const override { return no_default_index(); }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;

            protected:
                int64_t m_axis;
                uint64_t m_normalized_axis;
                Mode m_mode;
                SortType m_sort;
                element::Type m_index_element_type{element::i32};

                virtual size_t read_k_from_constant_node(const std::shared_ptr<Node>& node,
                                                         const element::Type& k_element_type) const;

                template <typename T>
                size_t validate_and_get_k(const std::shared_ptr<op::Constant>& k_constant) const;
                Shape compute_output_shape(const std::string& node_description,
                                           const PartialShape input_partial_shape,
                                           const int64_t k);
                void set_axis(const Rank input_rank, const int64_t axis);
            };
        } // namespace v1

        namespace v3
        {
            /// \brief Computes indices and values of the k maximum/minimum values
            ///        for each slice along specified axis.
            class NGRAPH_API TopK : public v1::TopK
            {
            public:
                static constexpr NodeTypeInfo type_info{"TopK", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a TopK operation
                TopK() = default;
                /// \brief Constructs a TopK operation with two outputs: values and indices.
                ///        By default the indices output is described by i32 data type.
                ///
                /// \param data The input tensor
                /// \param k Specifies how many maximum/minimum elements should be computed
                ///          (note: scalar input tensor)
                /// \param axis The axis along which to compute top k indices
                /// \param mode Specifies which operation (min or max) is used to select
                ///             the biggest element of two.
                /// \param sort Specifies order of output elements and/or indices
                ///             Accepted values: none, index, value
                /// \param index_element_type Specyfies type of produced indices
                TopK(const Output<Node>& data,
                     const Output<Node>& k,
                     const int64_t axis,
                     const std::string& mode,
                     const std::string& sort,
                     const element::Type& index_element_type = element::i32);

                TopK(const Output<Node>& data,
                     const Output<Node>& k,
                     const int64_t axis,
                     const Mode mode,
                     const SortType sort,
                     const element::Type& index_element_type = element::i32);
                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;

            protected:
                virtual size_t
                    read_k_from_constant_node(const std::shared_ptr<Node>& node,
                                              const element::Type& k_element_type) const override;
            };
        } // namespace v3

        using v0::TopK;
    } // op
} // ngraph
