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

#include <memory>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/validation_util.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/topk.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::TopK::type_info;

op::v0::TopK::TopK(const Output<Node>& arg,
                   size_t top_k_axis,
                   const element::Type& index_element_type,
                   size_t k,
                   bool compute_max,
                   SortType sort)
    : Op({arg})
    , m_index_element_type(index_element_type)
    , m_compute_max(compute_max)
    , m_sort(sort)
{
    set_argument(1, op::Constant::create(element::i64, Shape{1}, {k})->output(0));
    set_argument(2, op::Constant::create(element::i64, Shape{1}, {top_k_axis})->output(0));
    add_provenance_group_member(input_value(1).get_node_shared_ptr());
    add_provenance_group_member(input_value(2).get_node_shared_ptr());
    constructor_validate_and_infer_types();
}

op::v0::TopK::TopK(const Output<Node>& arg,
                   const Output<Node>& k,
                   size_t top_k_axis,
                   const element::Type& index_element_type,
                   bool compute_max,
                   SortType sort)
    : Op({arg, k})
    , m_index_element_type(index_element_type)
    , m_compute_max(compute_max)
    , m_sort(sort)
{
    set_argument(2, op::Constant::create(element::i64, Shape{1}, {top_k_axis})->output(0));
    add_provenance_group_member(input_value(2).get_node_shared_ptr());
    constructor_validate_and_infer_types();
}

op::v0::TopK::TopK(const Output<Node>& arg,
                   const Output<Node>& k,
                   const Output<Node>& top_k_axis,
                   const element::Type& index_element_type,
                   bool compute_max,
                   SortType sort)
    : Op({arg, k, top_k_axis})
    , m_index_element_type(index_element_type)
    , m_compute_max(compute_max)
    , m_sort(sort)
{
    constructor_validate_and_infer_types();
}

size_t op::v0::TopK::get_k() const
{
    size_t k = 0;
    if (auto const_op = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr()))
    {
        k = const_op->cast_vector<int64_t>()[0];
    }
    Dimension top_k_axis = get_top_k_axis_dynamic();
    if (k == 0 && get_input_partial_shape(0).is_static() && top_k_axis.is_static())
    {
        k = get_input_partial_shape(0).to_shape()[top_k_axis.get_length()];
    }
    return k;
}

void op::v0::TopK::set_k(size_t k)
{
    shared_ptr<Node> current_const =
        get_input_size() == 1 ? nullptr : input_value(1).get_node_shared_ptr();
    auto replacement_const = op::Constant::create(element::i64, Shape{1}, {k})->output(0);
    this->input(1).replace_source_output(replacement_const);
    replace_provenance_group_member(current_const, replacement_const.get_node_shared_ptr());
}

size_t op::v0::TopK::get_top_k_axis() const
{
    auto d = get_top_k_axis_dynamic();
    NGRAPH_CHECK(d.is_static(),
                 "get_top_k_axis called on a TopK node whose 'top_k_axis' input is not constant");
    return d.get_length();
}

Dimension op::v0::TopK::get_top_k_axis_dynamic() const
{
    auto const_op = dynamic_pointer_cast<op::Constant>(input_value(2).get_node_shared_ptr());
    if (const_op)
    {
        return const_op->cast_vector<int64_t>()[0];
    }
    else
    {
        return Dimension::dynamic();
    }
}

void op::v0::TopK::set_top_k_axis(size_t top_k_axis)
{
    shared_ptr<Node> current_const = input_value(2).get_node_shared_ptr();
    auto replacement_const = op::Constant::create(element::i64, Shape{1}, {top_k_axis})->output(0);
    this->input(2).replace_source_output(replacement_const);
    replace_provenance_group_member(current_const, replacement_const.get_node_shared_ptr());
}

void op::v0::TopK::validate_and_infer_types()
{
    const PartialShape& input_shape = get_input_partial_shape(0);
    Rank input_rank = input_shape.rank();
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(
        this, !m_index_element_type.is_dynamic(), "Argument element type must not be dynamic.");

    NODE_VALIDATION_CHECK(this,
                          m_index_element_type == element::i32 ||
                              m_index_element_type == element::i64,
                          "Argument element type must be i64 or i32 (got ",
                          m_index_element_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          input_rank.is_dynamic() || input_rank.get_length() > 0,
                          "Argument rank must be greater than 0.");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).compatible(element::i64),
                          "Element type for 'k' must be i64");
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2).compatible(element::i64),
                          "Element type for 'top_k_axis' must be i64");

    Dimension top_k_axis = get_top_k_axis_dynamic();
    NODE_VALIDATION_CHECK(this,
                          input_rank.is_dynamic() || top_k_axis.is_dynamic() ||
                              top_k_axis.get_length() < input_rank.get_length(),
                          "TopK axis (",
                          top_k_axis,
                          ") is out of bounds.");

    size_t k = get_k();
    NODE_VALIDATION_CHECK(this,
                          input_rank.is_dynamic() || top_k_axis.is_dynamic() ||
                              input_shape[top_k_axis.get_length()].is_dynamic() ||
                              static_cast<size_t>(k) <=
                                  input_shape[top_k_axis.get_length()].get_length(),
                          "K (",
                          k,
                          ") exceeds the dimension (",
                          input_shape[top_k_axis.get_length()],
                          ") of the TopK axis (axis ",
                          top_k_axis,
                          ").");

    PartialShape output_shape{input_shape};

    if (input_rank.is_static())
    {
        if (top_k_axis.is_static())
        {
            if (k != 0)
            {
                output_shape[top_k_axis.get_length()] = k;
            }
            else if (k == 0 && output_shape[top_k_axis.get_length()].is_static())
            {
                output_shape[top_k_axis.get_length()] = input_shape[top_k_axis.get_length()];
            }
        }
        else
        {
            // If top_k_axis is not static and k is not 0, then we could be changing any
            // dimension. So we have to change all dimensions to dynamic.
            output_shape = PartialShape::dynamic(input_rank);
        }
    }

    set_input_is_relevant_to_shape(2);

    set_output_size(2);
    set_output_type(0, m_index_element_type, output_shape);
    set_output_type(1, input_element_type, output_shape);
}

shared_ptr<Node> op::v0::TopK::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<TopK>(new_args.at(0),
                             new_args.at(1),
                             new_args.at(2),
                             m_index_element_type,
                             m_compute_max,
                             m_sort);
}

void op::v0::TopK::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                     const OutputVector& /* deltas */)
{
    throw ngraph_error("Forward-propagation-only operation");
}

namespace
{
    template <element::Type_t INPUT_ET, element::Type_t INDEX_ET>
    inline bool evaluate_execute(const HostTensorPtr& arg0,
                                 const HostTensorPtr& out_indices,
                                 const HostTensorPtr& out_values,
                                 const Shape out_shape,
                                 const size_t axis,
                                 const size_t k,
                                 const bool compute_max,
                                 const op::TopK::SortType sort)
    {
        using T = typename element_type_traits<INPUT_ET>::value_type;
        using U = typename element_type_traits<INDEX_ET>::value_type;
        const Shape in_shape = arg0->get_shape();
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

    template <element::Type_t INPUT_ET>
    bool evaluate(const HostTensorPtr& arg,
                  const HostTensorPtr& out_indices,
                  const HostTensorPtr& out_values,
                  const Shape out_shape,
                  const size_t axis,
                  const size_t k,
                  const bool max,
                  const op::TopK::SortType sort,
                  const element::Type index_et)
    {
        bool rc = true;
        switch (index_et)
        {
        case element::Type_t::i64:
            evaluate_execute<INPUT_ET, element::Type_t::i64>(
                arg, out_indices, out_values, out_shape, axis, k, max, sort);
            break;
        case element::Type_t::i32:
            evaluate_execute<INPUT_ET, element::Type_t::i32>(
                arg, out_indices, out_values, out_shape, axis, k, max, sort);
            break;
        default: rc = false; break;
        }
        return rc;
    }

    bool evaluate_topk(const HostTensorPtr& arg,
                       const HostTensorPtr& out_indices,
                       const HostTensorPtr& out_values,
                       const Shape out_shape,
                       const size_t axis,
                       const size_t k,
                       const bool max,
                       const op::TopK::SortType sort,
                       const element::Type index_et)
    {
        bool rc = true;
        switch (arg->get_element_type())
        {
            TYPE_CASE(i32)(arg, out_indices, out_values, out_shape, axis, k, max, sort, index_et);
            break;
            TYPE_CASE(i64)(arg, out_indices, out_values, out_shape, axis, k, max, sort, index_et);
            break;
            TYPE_CASE(u32)(arg, out_indices, out_values, out_shape, axis, k, max, sort, index_et);
            break;
            TYPE_CASE(u64)(arg, out_indices, out_values, out_shape, axis, k, max, sort, index_et);
            break;
            TYPE_CASE(f16)(arg, out_indices, out_values, out_shape, axis, k, max, sort, index_et);
            break;
            TYPE_CASE(f32)(arg, out_indices, out_values, out_shape, axis, k, max, sort, index_et);
            break;
        default: rc = false; break;
        }
        return rc;
    }

    template <element::Type_t K_ET>
    size_t get_k_from_hosttensor(const HostTensorPtr& arg)
    {
        using T = typename element_type_traits<K_ET>::value_type;
        auto p = arg->get_data_ptr<T>();
        size_t k = p[0];
        return k;
    }

#define CASE_GET_K(a)                                                                              \
    case element::Type_t::a: k = get_k_from_hosttensor<element::Type_t::a>

    size_t read_k_from_host_tensor(const HostTensorPtr& arg_k)
    {
        size_t k = 0;
        switch (arg_k->get_element_type())
        {
            CASE_GET_K(i8)(arg_k);
            break;
            CASE_GET_K(i16)(arg_k);
            break;
            CASE_GET_K(i32)(arg_k);
            break;
            CASE_GET_K(i64)(arg_k);
            break;
            CASE_GET_K(u8)(arg_k);
            break;
            CASE_GET_K(u16)(arg_k);
            break;
            CASE_GET_K(u32)(arg_k);
            break;
            CASE_GET_K(u64)(arg_k);
            break;
        default:
            // other types are not supported and would have thrown in ctor
            ngraph_error("read_k_from_host_tensor: type is not integral\n");
            break;
        }
        return k;
    }

    // used in only v0, where type is set as int64_t
    size_t read_top_k_axis_from_host_tensor(const HostTensorPtr& arg)
    {
        NGRAPH_CHECK(arg->get_element_type() == element::i64,
                     "TopK axis element type should be i64");
        auto p = arg->get_data_ptr<int64_t>();
        size_t axis = static_cast<size_t>(p[0]);
        return axis;
    }
}

Shape op::v0::TopK::compute_output_shape(const Shape input_shape,
                                         const int64_t k,
                                         const size_t axis)
{
    Shape output_shape{input_shape};
    if (k != 0)
    {
        output_shape[axis] = k;
    }
    return output_shape;
}

bool op::v0::TopK::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    // check data types for arg, k and output element type
    Shape arg_shape = inputs[0]->get_shape();

    // 1. get axis, mode ( max/min), sort_type
    size_t axis = 0;
    Dimension axis_dim = get_top_k_axis_dynamic();
    if (axis_dim.is_static())
    {
        axis = axis_dim.get_length();
    }
    else
    {
        axis = read_top_k_axis_from_host_tensor(inputs[2]);
        NGRAPH_CHECK(axis <= arg_shape.size(), "TopK axis is out of bounds");
    }
    bool compute_max = get_compute_max();
    SortType sort_type = get_sort();

    // 2. get value of k - from constant node or from HT
    size_t k = get_k();
    if (k == 0)
    {
        k = read_k_from_host_tensor(inputs[1]);
        if (k == 0)
        {
            // the kernel can't handle k = 0, but output_shape[axis] = arg_shape[axis]
            k = arg_shape[axis];
        }
    }
    NGRAPH_CHECK(k <= arg_shape.at(axis), "K exceeds the dimension of the TopK axis");

    // 3. Compute output_shape
    auto output_shape = compute_output_shape(inputs[0]->get_shape(), k, axis);

    return evaluate_topk(inputs[0],
                         outputs[0],
                         outputs[1],
                         output_shape,
                         axis,
                         k,
                         compute_max,
                         sort_type,
                         get_index_element_type());
}

// v1 version starts
constexpr NodeTypeInfo op::v1::TopK::type_info;

static const std::uint64_t UNKNOWN_NORMALIZED_AXIS = std::numeric_limits<uint64_t>::max();

op::v1::TopK::TopK(const Output<Node>& data,
                   const Output<Node>& k,
                   const int64_t axis,
                   const std::string& mode,
                   const std::string& sort,
                   const element::Type& index_element_type)
    : Op{{data, k}}
    , m_axis{axis}
    , m_normalized_axis{UNKNOWN_NORMALIZED_AXIS}
    , m_mode{as_enum<Mode>(mode)}
    , m_sort{as_enum<SortType>(sort)}
    , m_index_element_type{index_element_type}
{
    constructor_validate_and_infer_types();
}

op::v1::TopK::TopK(const Output<Node>& data,
                   const Output<Node>& k,
                   const int64_t axis,
                   const Mode mode,
                   const SortType sort,
                   const element::Type& index_element_type)
    : Op{{data, k}}
    , m_axis{axis}
    , m_normalized_axis{UNKNOWN_NORMALIZED_AXIS}
    , m_mode{mode}
    , m_sort{sort}
    , m_index_element_type{index_element_type}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::TopK::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("sort", m_sort);
    return true;
}

void op::v1::TopK::validate_and_infer_types()
{
    const auto& input_partial_shape = get_input_partial_shape(0);
    const auto input_rank = input_partial_shape.rank();

    NODE_VALIDATION_CHECK(this,
                          input_rank.is_dynamic() || input_rank.get_length() > 0,
                          "Input rank must be greater than 0.");

    const auto& k_partial_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(
        this, k_partial_shape.rank().compatible(0), "The 'K' input must be a scalar.");

    size_t k = 0;
    if (input_value(1).get_node_shared_ptr()->is_constant())
    {
        k = read_k_from_constant_node(input_value(1).get_node_shared_ptr(),
                                      get_input_element_type(1));
    }

    PartialShape output_shape{input_partial_shape};

    if (output_shape.rank().is_static())
    {
        m_normalized_axis = ngraph::normalize_axis(this, m_axis, output_shape.rank());
        if (k != 0)
        {
            output_shape[m_normalized_axis] = k;
        }
        else
        {
            auto max_k = maximum_value(input_value(1));
            if (max_k.first)
            {
                output_shape[m_normalized_axis] &= Dimension(0, max_k.second);
            }
            else
            {
                output_shape[m_normalized_axis] = -1;
            }
        }
    }

    set_output_size(2);
    set_output_type(0, get_input_element_type(0), output_shape);
    set_output_type(1, m_index_element_type, output_shape);
}

Shape op::v1::TopK::compute_output_shape(const std::string& node_description,
                                         const PartialShape input_partial_shape,
                                         const int64_t k)
{
    PartialShape output_shape{input_partial_shape};

    m_normalized_axis = ngraph::normalize_axis(node_description, m_axis, output_shape.rank());
    if (k != 0)
    {
        output_shape[m_normalized_axis] = k;
    }
    else
    {
        output_shape[m_normalized_axis] = input_partial_shape[m_normalized_axis];
    }

    return output_shape.get_shape();
}

void op::v1::TopK::set_axis(const int64_t axis)
{
    const auto input_rank = get_input_partial_shape(0).rank();
    if (input_rank.is_static())
    {
        m_normalized_axis = ngraph::normalize_axis(this, axis, input_rank);
    }
    else
    {
        m_normalized_axis = UNKNOWN_NORMALIZED_AXIS;
    }
    m_axis = axis;
}

void op::v1::TopK::set_axis(const Rank input_rank, const int64_t axis)
{
    if (input_rank.is_static())
    {
        m_normalized_axis = ngraph::normalize_axis(this, axis, input_rank);
    }
    else
    {
        m_normalized_axis = UNKNOWN_NORMALIZED_AXIS;
    }
    m_axis = axis;
}

uint64_t op::v1::TopK::get_axis() const
{
    NODE_VALIDATION_CHECK(
        this, m_normalized_axis != UNKNOWN_NORMALIZED_AXIS, "Normalized axis of TopK is unknown");

    return m_normalized_axis;
}

size_t op::v1::TopK::read_k_from_constant_node(const shared_ptr<Node>& node,
                                               const element::Type& k_element_type) const
{
    NODE_VALIDATION_CHECK(this,
                          k_element_type == element::i8 || k_element_type == element::i32 ||
                              k_element_type == element::i64,
                          "K input element type must be i8, i32 or i64 (got ",
                          k_element_type,
                          ").");

    const auto k_constant = as_type_ptr<op::Constant>(node);

    size_t k = 0;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
#endif
    switch (static_cast<element::Type_t>(k_element_type))
    {
    case element::Type_t::i8: k = validate_and_get_k<int8_t>(k_constant); break;
    case element::Type_t::i32: k = validate_and_get_k<int32_t>(k_constant); break;
    case element::Type_t::i64: k = validate_and_get_k<int64_t>(k_constant); break;
    default: break;
    }
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    return k;
}

template <typename T>
size_t op::v1::TopK::validate_and_get_k(const shared_ptr<op::Constant>& k_constant) const
{
    const auto k_const_contents = k_constant->get_vector<T>();

    NODE_VALIDATION_CHECK(this,
                          k_const_contents.size() == 1,
                          "Only one value (scalar) should be provided as the 'K' input to TopK",
                          " (got ",
                          k_const_contents.size(),
                          " elements).");

    NODE_VALIDATION_CHECK(this,
                          k_const_contents[0] > 0,
                          "The value of 'K' must be a positive number.",
                          " (got ",
                          k_const_contents[0],
                          ").");

    return static_cast<size_t>(k_const_contents[0]);
}

void op::v1::TopK::generate_adjoints(autodiff::Adjoints& /*adjoints*/,
                                     const OutputVector& /* deltas */)
{
    throw ngraph_error("Forward-propagation-only operation");
}

shared_ptr<Node> op::v1::TopK::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    auto new_v1_topk =
        make_shared<v1::TopK>(new_args.at(0), new_args.at(1), m_axis, m_mode, m_sort);

    new_v1_topk->set_index_element_type(m_index_element_type);

    return std::move(new_v1_topk);
}

size_t op::v1::TopK::get_k() const
{
    size_t k = 0;
    if (input_value(1).get_node_shared_ptr()->is_constant())
    {
        k = read_k_from_constant_node(input_value(1).get_node_shared_ptr(),
                                      get_input_element_type(1));
    }

    if (k == 0 && get_input_partial_shape(0).is_static())
    {
        k = get_input_partial_shape(0).to_shape()[m_normalized_axis];
    }
    return k;
}

void op::v1::TopK::set_k(size_t k)
{
    this->input(1).replace_source_output(
        op::Constant::create(element::i64, Shape{}, {k})->output(0));
}

bool op::v1::TopK::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    Shape arg_shape = inputs[0]->get_shape();
    // 1. get axis, mode ( max/min), sort_type
    set_axis(arg_shape.size(), m_axis);
    size_t axis = get_axis();
    bool compute_max = get_mode() == TopKMode::MAX ? true : false;
    SortType sort_type = get_sort_type();

    // 2. get value of k - from constant node or from HT
    size_t k = 0;
    if (input_value(1).get_node_shared_ptr()->is_constant())
    {
        k = read_k_from_constant_node(input_value(1).get_node_shared_ptr(),
                                      get_input_element_type(1));
        NGRAPH_CHECK(k <= arg_shape[axis], "'K' exceeds the dimension of top_k_axis");
    }
    else
    {
        k = read_k_from_host_tensor(inputs[1]);
    }

    // 3. Compute output_shape
    auto output_shape = compute_output_shape(this->description(), inputs[0]->get_shape(), k);

    // do this after compute_output_shape
    if (k == 0)
    {
        // the kernel can't handle k = 0, but output_shape[axis] = arg_shape[axis]
        k = arg_shape[axis];
    }

    return evaluate_topk(inputs[0],
                         outputs[1],
                         outputs[0],
                         output_shape,
                         axis,
                         k,
                         compute_max,
                         sort_type,
                         get_index_element_type());
}

// v3 version starts
constexpr NodeTypeInfo op::v3::TopK::type_info;

op::v3::TopK::TopK(const Output<Node>& data,
                   const Output<Node>& k,
                   const int64_t axis,
                   const std::string& mode,
                   const std::string& sort,
                   const element::Type& index_element_type)
    : op::v1::TopK{data, k, axis, mode, sort, index_element_type}
{
    constructor_validate_and_infer_types();
}

op::v3::TopK::TopK(const Output<Node>& data,
                   const Output<Node>& k,
                   const int64_t axis,
                   const Mode mode,
                   const SortType sort,
                   const element::Type& index_element_type)
    : op::v1::TopK{data, k, axis, mode, sort, index_element_type}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v3::TopK::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("sort", m_sort);
    visitor.on_attribute("index_element_type", m_index_element_type);
    return true;
}

void op::v3::TopK::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number(),
                          "K input has to be an integer type, which does match the provided one:",
                          get_input_element_type(1));
    op::v1::TopK::validate_and_infer_types();
}

size_t op::v3::TopK::read_k_from_constant_node(const shared_ptr<Node>& node,
                                               const element::Type& k_element_type) const
{
    const auto k_constant = as_type_ptr<op::Constant>(node);

    size_t k = 0;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
#endif
    switch (static_cast<element::Type_t>(k_element_type))
    {
    case element::Type_t::i8: k = validate_and_get_k<int8_t>(k_constant); break;
    case element::Type_t::i16: k = validate_and_get_k<int16_t>(k_constant); break;
    case element::Type_t::i32: k = validate_and_get_k<int32_t>(k_constant); break;
    case element::Type_t::i64: k = validate_and_get_k<int64_t>(k_constant); break;
    case element::Type_t::u8: k = validate_and_get_k<uint8_t>(k_constant); break;
    case element::Type_t::u16: k = validate_and_get_k<uint16_t>(k_constant); break;
    case element::Type_t::u32: k = validate_and_get_k<uint32_t>(k_constant); break;
    case element::Type_t::u64: k = validate_and_get_k<uint64_t>(k_constant); break;
    default: break;
    }
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    return k;
}

shared_ptr<Node> op::v3::TopK::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    auto new_v3_topk =
        make_shared<v3::TopK>(new_args.at(0), new_args.at(1), m_axis, m_mode, m_sort);

    new_v3_topk->set_index_element_type(m_index_element_type);

    return std::move(new_v3_topk);
}

bool op::v3::TopK::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    return op::v1::TopK::evaluate(outputs, inputs);
}
