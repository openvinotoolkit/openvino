// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/pad.hpp"
#include <ngraph/validation_util.hpp>
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/pad.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::Pad::type_info;

op::v1::Pad::Pad(const Output<Node>& arg,
                 const Output<Node>& pads_begin,
                 const Output<Node>& pads_end,
                 const Output<Node>& arg_pad_value,
                 PadMode pad_mode)
    : Op({arg, pads_begin, pads_end, arg_pad_value})
    , m_pad_mode{pad_mode}
{
    constructor_validate_and_infer_types();
}

op::v1::Pad::Pad(const Output<Node>& arg,
                 const Output<Node>& pads_begin,
                 const Output<Node>& pads_end,
                 PadMode pad_mode)
    : Op({arg, pads_begin, pads_end, op::Constant::create(arg.get_element_type(), Shape{}, {0})})
    , m_pad_mode{pad_mode}
{
    constructor_validate_and_infer_types();
}

CoordinateDiff op::v1::Pad::get_pads_begin() const
{
    CoordinateDiff pads_begin_coord{};
    if (auto pads_begin_const = get_constant_from_source(input_value(1)))
    {
        pads_begin_coord = pads_begin_const->cast_vector<ptrdiff_t>();
    }
    return pads_begin_coord;
}

CoordinateDiff op::v1::Pad::get_pads_end() const
{
    CoordinateDiff pads_end_coord{};
    if (auto pads_end_const = get_constant_from_source(input_value(2)))
    {
        pads_end_coord = pads_end_const->cast_vector<ptrdiff_t>();
    }
    return pads_end_coord;
}

bool ngraph::op::v1::Pad::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_Pad_visit_attributes);
    visitor.on_attribute("pad_mode", m_pad_mode);
    return true;
}

void op::v1::Pad::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_Pad_validate_and_infer_types);
    element::Type result_et;

    const auto& arg_element_type = get_input_element_type(0);
    const auto& pads_begin_element_type = get_input_element_type(1);
    const auto& pads_end_element_type = get_input_element_type(2);

    if (m_pad_mode == PadMode::CONSTANT && get_input_size() == 4)
    {
        const auto& arg_pad_element_type = get_input_element_type(3);
        const auto& arg_pad_shape = get_input_partial_shape(3);
        NODE_VALIDATION_CHECK(
            this,
            element::Type::merge(result_et, arg_element_type, arg_pad_element_type),
            "Argument element types do not match (input arg element type: ",
            arg_element_type,
            ", arg_pad element type: ",
            arg_pad_element_type,
            ").");

        NODE_VALIDATION_CHECK(this,
                              arg_pad_shape.compatible(PartialShape{}),
                              "Argument for padding value is not a scalar (shape: ",
                              arg_pad_shape,
                              ").");
    }

    NODE_VALIDATION_CHECK(this,
                          pads_begin_element_type.is_integral_number(),
                          "pads_begin must be an integral number, but is: ",
                          pads_begin_element_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          pads_end_element_type.is_integral_number(),
                          "pads_end must be an integral number, but is: ",
                          pads_end_element_type,
                          ").");

    const auto& pads_begin_shape = get_input_partial_shape(1);
    const auto& pads_begin_rank = pads_begin_shape.rank();

    NODE_VALIDATION_CHECK(this,
                          pads_begin_rank.compatible(1),
                          "Argument for pads_begin is not 1D (shape: ",
                          pads_begin_rank,
                          ").");

    const auto& pads_end_shape = get_input_partial_shape(2);
    const auto& pads_end_rank = pads_end_shape.rank();
    NODE_VALIDATION_CHECK(this,
                          pads_end_rank.compatible(1),
                          "Argument for pads_end is not 1D (shape: ",
                          pads_end_rank,
                          ").");

    const auto& arg_shape = get_input_partial_shape(0);
    const auto& arg_shape_rank = arg_shape.rank();
    if (arg_shape_rank.is_static() && pads_begin_shape.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              pads_begin_shape[0].get_length() <= arg_shape_rank.get_length(),
                              "Number of elements of pads_begin must be >= 0 and <= arg rank "
                              "(pads_begin_shape[0]: ",
                              pads_begin_shape[0],
                              ").");
    }
    if (arg_shape_rank.is_static() && pads_end_shape.is_static())
    {
        NODE_VALIDATION_CHECK(
            this,
            pads_end_shape[0].get_length() <= arg_shape_rank.get_length(),
            "Number of elements of pads_end must be >= 0 and <= arg rank (pads_end_shape[0]: ",
            pads_end_shape[0],
            ").");
    }
    const auto& pads_begin_coord = get_pads_begin();
    const auto& pads_end_coord = get_pads_end();

    if (arg_shape_rank.is_static() && !pads_begin_coord.empty() && !pads_end_coord.empty())
    {
        const auto implied_rank = pads_begin_coord.size();
        std::vector<Dimension> result_dims(implied_rank, Dimension::dynamic());
        for (size_t i = 0; i < implied_rank; i++)
        {
            if (arg_shape[i].is_static())
            {
                ptrdiff_t result_dim =
                    pads_begin_coord[i] + arg_shape[i].get_length() + pads_end_coord[i];
                result_dims[i] = static_cast<size_t>(result_dim);
                if (i > 1)
                {
                    NODE_VALIDATION_CHECK(this,
                                          m_pad_mode != op::PadMode::EDGE ||
                                              arg_shape[i].get_length() >= 1,
                                          "EDGE padding mode requires an input of dimension of "
                                          "at least 1 at each "
                                          "spatial axis.");
                    NODE_VALIDATION_CHECK(this,
                                          m_pad_mode != op::PadMode::REFLECT ||
                                              arg_shape[i].get_length() >= 2,
                                          "REFLECT padding mode requires an input of dimension "
                                          "of at least 2 at each "
                                          "spatial axis.");
                }
            }
        }
        set_output_type(0, get_input_element_type(0), result_dims);
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic(arg_shape_rank));
    }
}

shared_ptr<Node> op::v1::Pad::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Pad_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (get_input_size() == 4)
    {
        return make_shared<v1::Pad>(
            new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_pad_mode);
    }
    else
    {
        return make_shared<v1::Pad>(new_args.at(0), new_args.at(1), new_args.at(2), m_pad_mode);
    }
}

bool op::v1::Pad::evaluate_pad(const HostTensorVector& outputs,
                               const HostTensorVector& inputs) const
{
    const auto& data = inputs[0];
    const auto elem_size = data->get_element_type().size();

    const char* pad_value = nullptr;
    const std::vector<char> pad_zero_value(elem_size, 0);
    if (get_input_size() == 4)
    {
        pad_value = inputs[3]->get_data_ptr<char>();
    }
    else
    {
        pad_value = pad_zero_value.data();
    }
    const auto& out = outputs[0];

    ngraph::runtime::reference::pad(data->get_data_ptr<char>(),
                                    pad_value,
                                    out->get_data_ptr<char>(),
                                    elem_size,
                                    data->get_shape(),
                                    out->get_shape(),
                                    get_pads_begin(),
                                    get_pads_end(),
                                    get_pad_mode());

    return true;
}

bool op::v1::Pad::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Pad_evaluate);
    return evaluate_pad(outputs, inputs);
}

bool op::v1::Pad::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_Pad_has_evaluate);
    return true;
}
