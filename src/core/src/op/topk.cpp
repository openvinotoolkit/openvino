// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/topk.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/topk.hpp"
#include "topk_shape_inference.hpp"

namespace ov {
namespace op {
namespace topk {
namespace validate {
namespace {
bool data_type(const element::Type& et) {
    switch (et) {
    case element::f16:
    case element::f32:
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}

bool k_type(const element::Type& et) {
    switch (et) {
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}
}  // namespace
}  // namespace validate

struct Evaluate : public element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in,
                             Tensor& out_values,
                             Tensor& out_indices,
                             const Shape& out_shape,
                             const size_t axis,
                             const bool compute_max,
                             const TopKSortType sort) {
        using namespace ov::element;
        return IF_TYPE_OF(topk_eval_by_idx_type,
                          OV_PP_ET_LIST(i32, i64),
                          EvalByIdxType,
                          out_indices.get_element_type(),
                          in.data<const T>(),
                          out_values.data<T>(),
                          out_indices,
                          in.get_shape(),
                          out_shape,
                          axis,
                          out_shape[axis],
                          compute_max,
                          sort);
    }

private:
    struct EvalByIdxType : public element::NoAction<bool> {
        using element::NoAction<bool>::visit;

        template <element::Type_t ET, class T, class I = fundamental_type_for<ET>>
        static result_type visit(const T* in_first,
                                 T* out_first,
                                 Tensor& out_indices,
                                 const Shape& in_shape,
                                 const Shape& out_shape,
                                 const size_t axis,
                                 const size_t k,
                                 const bool compute_max,
                                 const TopKSortType sort) {
            reference::topk(in_first,
                            out_indices.data<I>(),
                            out_first,
                            in_shape,
                            out_shape,
                            axis,
                            k,
                            compute_max,
                            sort);
            return true;
        }
    };
};

namespace {
bool evaluate(const util::TopKBase* const node, TensorVector& outputs, const TensorVector& inputs) {
    auto output_shapes = shape_infer(node, ov::util::get_tensors_partial_shapes(inputs), make_tensor_accessor(inputs));
    OPENVINO_ASSERT(outputs.size() == output_shapes.size());

    auto output_shape = output_shapes.front().get_shape();
    const auto axis = ov::util::normalize(node->get_provided_axis(), output_shape.size());
    if (output_shape[axis] == 0) {
        // the kernel can't handle K (output_shape[axis]) equal 0, use arg_shape[axis] instead.
        output_shape[axis] = inputs[0].get_shape()[axis];
    }

    for (auto& t : outputs) {
        t.set_shape(output_shape);
    }

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(topk_evaluate,
                                      node,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64, u32, u64),
                                      topk::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      outputs[1],
                                      output_shape,
                                      axis,
                                      (node->get_mode() == ov::op::TopKMode::MAX),
                                      node->get_sort_type());
}
}  // namespace
}  // namespace topk

// v1 version starts
namespace v1 {
TopK::TopK(const Output<Node>& data,
           const Output<Node>& k,
           const int64_t axis,
           const std::string& mode,
           const std::string& sort,
           const element::Type& index_element_type)
    : util::TopKBase(data, k, axis, mode, sort, index_element_type) {
    constructor_validate_and_infer_types();
}

TopK::TopK(const Output<Node>& data,
           const Output<Node>& k,
           const int64_t axis,
           const Mode mode,
           const SortType sort,
           const element::Type& index_element_type)
    : util::TopKBase(data, k, axis, mode, sort, index_element_type) {
    constructor_validate_and_infer_types();
}

void TopK::k_type_check(const element::Type& k_element_type) const {
    NODE_VALIDATION_CHECK(
        this,
        k_element_type == element::i8 || k_element_type == element::i32 || k_element_type == element::i64,
        "K input element type must be i8, i32 or i64 (got ",
        k_element_type,
        ").");
}

std::shared_ptr<Node> TopK::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_TopK_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<TopK>(new_args.at(0), new_args.at(1), m_axis, m_mode, m_sort, m_index_element_type);
}

bool TopK::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_TopK_evaluate);
    return topk::evaluate(this, outputs, inputs);
}

bool TopK::has_evaluate() const {
    OV_OP_SCOPE(v1_TopK_has_evaluate);
    return topk::validate::data_type(get_input_element_type(0)) && topk::validate::k_type(get_input_element_type(1));
}
}  // namespace v1

// v3 version starts
namespace v3 {
TopK::TopK(const Output<Node>& data,
           const Output<Node>& k,
           const int64_t axis,
           const std::string& mode,
           const std::string& sort,
           const element::Type& index_element_type)
    : TopK(data, k, axis, as_enum<Mode>(mode), as_enum<SortType>(sort), index_element_type) {}

TopK::TopK(const Output<Node>& data,
           const Output<Node>& k,
           const int64_t axis,
           const Mode mode,
           const SortType sort,
           const element::Type& index_element_type)
    : util::TopKBase{data, k, axis, mode, sort, index_element_type} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> TopK::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_TopK_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<TopK>(new_args.at(0), new_args.at(1), m_axis, m_mode, m_sort, m_index_element_type);
}

bool TopK::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v3_TopK_evaluate);
    return topk::evaluate(this, outputs, inputs);
}

bool TopK::has_evaluate() const {
    OV_OP_SCOPE(v3_TopK_has_evaluate);
    return topk::validate::data_type(get_input_element_type(0)) && topk::validate::k_type(get_input_element_type(1));
}
}  // namespace v3

// =============== V11 ===============
namespace v11 {
TopK::TopK(const Output<Node>& data,
           const Output<Node>& k,
           const int64_t axis,
           const std::string& mode,
           const std::string& sort,
           const element::Type& index_element_type,
           const bool stable)
    : TopK(data, k, axis, as_enum<TopKMode>(mode), as_enum<TopKSortType>(sort), index_element_type, stable) {}

TopK::TopK(const Output<Node>& data,
           const Output<Node>& k,
           const int64_t axis,
           const TopKMode mode,
           const TopKSortType sort,
           const element::Type& index_element_type,
           const bool stable)
    : util::TopKBase{data, k, axis, mode, sort, index_element_type},
      m_stable{stable} {
    constructor_validate_and_infer_types();
}

void TopK::validate_and_infer_types() {
    OV_OP_SCOPE(v11_TopK_validate_and_infer_types);

    if (m_stable) {
        NODE_VALIDATION_CHECK(this,
                              m_sort == TopKSortType::SORT_VALUES || m_sort == TopKSortType::SORT_INDICES,
                              "Stable sort can only be used when TopK's sorting mode is set to 'VALUE' or 'INDEX'.",
                              AttributeAdapter<TopKSortType>(m_sort).get());
    }

    util::TopKBase::validate_and_infer_types();
}

bool TopK::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v11_TopK_visit_attributes);
    util::TopKBase::visit_attributes(visitor);
    visitor.on_attribute("stable", m_stable);
    return true;
}

std::shared_ptr<Node> TopK::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v11_TopK_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<TopK>(new_args.at(0),
                                  new_args.at(1),
                                  m_axis,
                                  m_mode,
                                  m_sort,
                                  m_index_element_type,
                                  m_stable);
}

bool TopK::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v11_TopK_evaluate);
    return topk::evaluate(this, outputs, inputs);
}

bool TopK::has_evaluate() const {
    OV_OP_SCOPE(v11_TopK_has_evaluate);
    return topk::validate::data_type(get_input_element_type(0));
}
}  // namespace v11
}  // namespace op
}  // namespace ov
