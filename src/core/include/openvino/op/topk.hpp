// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/constant.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Computes indices and values of the k maximum/minimum values
///        for each slice along specified axis.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API TopK : public Op {
public:
    OPENVINO_OP("TopK", "opset1", op::Op, 1);
    BWDCMP_RTTI_DECLARATION;

    using SortType = TopKSortType;
    using Mode = TopKMode;

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

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \brief Returns axis value after normalization
    /// \note If input rank required to normalization is dynamic, the exception is
    /// thrown
    uint64_t get_axis() const;
    /// \brief Returns axis value before normalization
    int64_t get_provided_axis() const {
        return m_axis;
    }
    void set_axis(const int64_t axis);
    Mode get_mode() const {
        return m_mode;
    }
    void set_mode(const Mode mode) {
        m_mode = mode;
    }
    SortType get_sort_type() const {
        return m_sort;
    }
    void set_sort_type(const SortType sort) {
        m_sort = sort;
    }
    element::Type get_index_element_type() const {
        return m_index_element_type;
    }
    void set_index_element_type(const element::Type& index_element_type) {
        m_index_element_type = index_element_type;
    }
    /// \brief Returns the value of K, if available
    ///
    /// \note If the second input to this op is a constant, the value is retrieved
    ///       and returned. If the input is not constant(dynamic) this method returns 0
    size_t get_k() const;
    void set_k(size_t k);
    size_t get_default_output_index() const override {
        return no_default_index();
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;

protected:
    int64_t m_axis;
    uint64_t m_normalized_axis;
    Mode m_mode;
    SortType m_sort;
    element::Type m_index_element_type{element::i32};

    virtual size_t read_k_from_constant_node(const std::shared_ptr<Node>& node,
                                             const element::Type& k_element_type) const;

    template <typename T>
    size_t validate_and_get_k(const std::shared_ptr<op::v0::Constant>& k_constant) const;
    Shape compute_output_shape(const std::string& node_description,
                               const PartialShape input_partial_shape,
                               const int64_t k) const;
    void set_axis(const Rank input_rank, const int64_t axis);
};
}  // namespace v1

namespace v3 {
/// \brief Computes indices and values of the k maximum/minimum values
///        for each slice along specified axis.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API TopK : public v1::TopK {
public:
    OPENVINO_OP("TopK", "opset3", op::Op, 3);
    BWDCMP_RTTI_DECLARATION;
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
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;

protected:
    size_t read_k_from_constant_node(const std::shared_ptr<Node>& node,
                                     const element::Type& k_element_type) const override;
};
}  // namespace v3
}  // namespace op
}  // namespace ov
