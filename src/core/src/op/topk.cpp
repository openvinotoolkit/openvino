// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/topk.hpp"

#include <memory>
#include <topk_shape_inference.hpp>

#include "dimension_tracker.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/topk.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

namespace topk {
namespace {
template <element::Type_t INPUT_ET, element::Type_t INDEX_ET>
inline bool evaluate_execute(const HostTensorPtr& arg0,
                             const HostTensorPtr& out_indices,
                             const HostTensorPtr& out_values,
                             const ov::Shape out_shape,
                             const size_t axis,
                             const size_t k,
                             const bool compute_max,
                             const op::v1::TopK::SortType sort) {
    using T = typename element_type_traits<INPUT_ET>::value_type;
    using U = typename element_type_traits<INDEX_ET>::value_type;
    const ov::Shape in_shape = arg0->get_shape();
    out_indices->set_shape(out_shape);
    out_indices->set_element_type(INDEX_ET);

    out_values->set_shape(out_shape);
    out_values->set_element_type(arg0->get_element_type());

    runtime::reference::topk<T, U>(arg0->get_data_ptr<INPUT_ET>(),
                                   out_indices->get_data_ptr<INDEX_ET>(),
                                   out_values->get_data_ptr<INPUT_ET>(),
                                   in_shape,
                                   out_shape,
                                   axis,
                                   k,
                                   compute_max,
                                   sort);
    return true;
}

#define EXECUTE_EVALUATE_TOPK(a, ...)                                     \
    case element::Type_t::a: {                                            \
        OV_OP_SCOPE(OV_PP_CAT3(exec_topk_eval, _, a));                    \
        rc = evaluate_execute<INPUT_ET, element::Type_t::a>(__VA_ARGS__); \
    } break

template <element::Type_t INPUT_ET>
bool evaluate(const HostTensorPtr& arg,
              const HostTensorPtr& out_indices,
              const HostTensorPtr& out_values,
              const ov::Shape out_shape,
              const size_t axis,
              const size_t k,
              const bool max,
              const op::v1::TopK::SortType sort,
              const element::Type index_et) {
    bool rc = true;
    switch (index_et) {
        EXECUTE_EVALUATE_TOPK(i32, arg, out_indices, out_values, out_shape, axis, k, max, sort);
        EXECUTE_EVALUATE_TOPK(i64, arg, out_indices, out_values, out_shape, axis, k, max, sort);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool evaluate_topk(const HostTensorPtr& arg,
                   const HostTensorPtr& out_indices,
                   const HostTensorPtr& out_values,
                   const ov::Shape out_shape,
                   const size_t axis,
                   const size_t k,
                   const bool max,
                   const op::v1::TopK::SortType sort,
                   const element::Type index_et) {
    bool rc = true;
    switch (arg->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_topk, i32, arg, out_indices, out_values, out_shape, axis, k, max, sort, index_et);
        NGRAPH_TYPE_CASE(evaluate_topk, i64, arg, out_indices, out_values, out_shape, axis, k, max, sort, index_et);
        NGRAPH_TYPE_CASE(evaluate_topk, u32, arg, out_indices, out_values, out_shape, axis, k, max, sort, index_et);
        NGRAPH_TYPE_CASE(evaluate_topk, u64, arg, out_indices, out_values, out_shape, axis, k, max, sort, index_et);
        NGRAPH_TYPE_CASE(evaluate_topk, f16, arg, out_indices, out_values, out_shape, axis, k, max, sort, index_et);
        NGRAPH_TYPE_CASE(evaluate_topk, f32, arg, out_indices, out_values, out_shape, axis, k, max, sort, index_et);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace topk

// v1 version starts

op::v1::TopK::TopK(const Output<Node>& data,
                   const Output<Node>& k,
                   const int64_t axis,
                   const std::string& mode,
                   const std::string& sort,
                   const element::Type& index_element_type)
    : util::TopKBase(data, k, axis, mode, sort, index_element_type) {
    constructor_validate_and_infer_types();
}

op::v1::TopK::TopK(const Output<Node>& data,
                   const Output<Node>& k,
                   const int64_t axis,
                   const Mode mode,
                   const SortType sort,
                   const element::Type& index_element_type)
    : util::TopKBase(data, k, axis, mode, sort, index_element_type) {
    constructor_validate_and_infer_types();
}

void op::v1::TopK::k_type_check(const element::Type& k_element_type) const {
    NODE_VALIDATION_CHECK(
        this,
        k_element_type == element::i8 || k_element_type == element::i32 || k_element_type == element::i64,
        "K input element type must be i8, i32 or i64 (got ",
        k_element_type,
        ").");
}

shared_ptr<Node> op::v1::TopK::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_TopK_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::TopK>(new_args.at(0), new_args.at(1), m_axis, m_mode, m_sort, m_index_element_type);
}

bool op::v1::TopK::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_TopK_evaluate);
    const auto& arg_shape = inputs[0]->get_shape();
    // 1. get axis, mode (max/min), sort_type
    auto axis = ngraph::normalize_axis(this, m_axis, arg_shape.size());
    auto compute_max = get_mode() == TopKMode::MAX;
    auto sort_type = get_sort_type();

    const auto input_shapes = std::vector<PartialShape>{inputs[0]->get_partial_shape(), inputs[1]->get_partial_shape()};
    const auto constant_data = std::map<size_t, HostTensorPtr>{{1, inputs[1]}};
    auto output_shape = shape_infer(this, input_shapes, constant_data).front().to_shape();

    if (output_shape[axis] == 0) {
        // the kernel can't handle K (output_shape[axis]) equal 0, use arg_shape[axis] instead.
        output_shape[axis] = arg_shape[axis];
    }

    // 2. get value of k
    size_t k = output_shape[axis];
    OPENVINO_ASSERT(k <= arg_shape[axis], "'K' exceeds the dimension of top_k_axis");

    return topk::evaluate_topk(inputs[0],
                               outputs[1],
                               outputs[0],
                               output_shape,
                               axis,
                               k,
                               compute_max,
                               sort_type,
                               get_index_element_type());
}

bool op::v1::TopK::has_evaluate() const {
    OV_OP_SCOPE(v1_TopK_has_evaluate);

    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
        break;
    default:
        return false;
    }

    if (op::util::is_constant(input_value(1).get_node())) {
        switch (get_input_element_type(1)) {
        case ngraph::element::i8:
        case ngraph::element::i32:
        case ngraph::element::i64:
            break;
        default:
            return false;
        }
    } else {
        switch (get_input_element_type(1)) {
        case ngraph::element::i8:
        case ngraph::element::i16:
        case ngraph::element::i32:
        case ngraph::element::i64:
        case ngraph::element::u8:
        case ngraph::element::u16:
        case ngraph::element::u32:
        case ngraph::element::u64:
            break;
        default:
            return false;
        }
    }

    return true;
}

// v3 version starts
op::v3::TopK::TopK(const Output<Node>& data,
                   const Output<Node>& k,
                   const int64_t axis,
                   const std::string& mode,
                   const std::string& sort,
                   const element::Type& index_element_type)
    : TopK(data, k, axis, as_enum<Mode>(mode), as_enum<SortType>(sort), index_element_type) {}

op::v3::TopK::TopK(const Output<Node>& data,
                   const Output<Node>& k,
                   const int64_t axis,
                   const Mode mode,
                   const SortType sort,
                   const element::Type& index_element_type)
    : util::TopKBase{data, k, axis, mode, sort, index_element_type} {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v3::TopK::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_TopK_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v3::TopK>(new_args.at(0), new_args.at(1), m_axis, m_mode, m_sort, m_index_element_type);
}

bool op::v3::TopK::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v3_TopK_evaluate);
    const auto& arg_shape = inputs[0]->get_shape();
    // 1. get axis, mode (max/min), sort_type
    auto axis = ngraph::normalize_axis(this, m_axis, arg_shape.size());
    auto compute_max = get_mode() == TopKMode::MAX;
    auto sort_type = get_sort_type();

    const auto input_shapes = std::vector<PartialShape>{inputs[0]->get_partial_shape(), inputs[1]->get_partial_shape()};
    const auto constant_data = std::map<size_t, HostTensorPtr>{{1, inputs[1]}};
    auto output_shape = shape_infer(this, input_shapes, constant_data).front().to_shape();

    if (output_shape[axis] == 0) {
        // the kernel can't handle K (output_shape[axis]) equal 0, use arg_shape[axis] instead.
        output_shape[axis] = arg_shape[axis];
    }

    // 2. get value of k
    size_t k = output_shape[axis];
    OPENVINO_ASSERT(k <= arg_shape[axis], "'K' exceeds the dimension of top_k_axis");

    return topk::evaluate_topk(inputs[0],
                               outputs[1],
                               outputs[0],
                               output_shape,
                               axis,
                               k,
                               compute_max,
                               sort_type,
                               get_index_element_type());
}

bool op::v3::TopK::has_evaluate() const {
    OV_OP_SCOPE(v3_TopK_has_evaluate);

    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
        break;
    default:
        return false;
    }

    if (op::util::is_constant(input_value(1).get_node())) {
        switch (get_input_element_type(1)) {
        case ngraph::element::i8:
        case ngraph::element::i32:
        case ngraph::element::i64:
            break;
        default:
            return false;
        }
    } else {
        switch (get_input_element_type(1)) {
        case ngraph::element::i8:
        case ngraph::element::i16:
        case ngraph::element::i32:
        case ngraph::element::i64:
        case ngraph::element::u8:
        case ngraph::element::u16:
        case ngraph::element::u32:
        case ngraph::element::u64:
            break;
        default:
            return false;
        }
    }

    return true;
}

// =============== V11 ===============
ov::op::v11::TopK::TopK(const Output<Node>& data,
                        const Output<Node>& k,
                        const int64_t axis,
                        const std::string& mode,
                        const std::string& sort,
                        const element::Type& index_element_type,
                        const bool stable)
    : TopK(data, k, axis, as_enum<TopKMode>(mode), as_enum<TopKSortType>(sort), index_element_type, stable) {}

ov::op::v11::TopK::TopK(const Output<Node>& data,
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

void ov::op::v11::TopK::validate_and_infer_types() {
    OV_OP_SCOPE(v11_TopK_validate_and_infer_types);

    if (m_stable) {
        NODE_VALIDATION_CHECK(
            this,
            m_sort == TopKSortType::SORT_VALUES,
            "Stable sort can only be used when TopK's sorting mode is set to 'VALUE'. Current sorting mode = ",
            AttributeAdapter<TopKSortType>(m_sort).get());
    }

    util::TopKBase::validate_and_infer_types();
}

bool ov::op::v11::TopK::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v11_TopK_visit_attributes);
    util::TopKBase::visit_attributes(visitor);
    visitor.on_attribute("stable", m_stable);
    return true;
}

std::shared_ptr<Node> ov::op::v11::TopK::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v11_TopK_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ov::op::v11::TopK>(new_args.at(0),
                                          new_args.at(1),
                                          m_axis,
                                          m_mode,
                                          m_sort,
                                          m_index_element_type,
                                          m_stable);
}
