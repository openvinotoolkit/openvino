// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/tile.hpp"
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/reference/tile.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::Tile::type_info;

op::v0::Tile::Tile(const Output<Node>& data, const Output<Node>& repeats)
    : Op({data, repeats})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Tile::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Tile_visit_attributes);
    return true;
}

void op::v0::Tile::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_Tile_validate_and_infer_types);
    auto arg_et = get_input_element_type(0);

    // Repeats should have integer data type. For now we only allow i64
    auto repeats_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          repeats_et.is_integral(),
                          "Tile repeats must have any integer element type, but has ",
                          repeats_et);

    auto arg_shape = get_input_partial_shape(0);
    auto repeats_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(
        this, repeats_shape.rank().compatible(1), "Shape of repeats must be of rank 1");
    PartialShape repeats_as_pshape;
    bool repeats_are_known =
        evaluate_as_partial_shape(get_input_source_output(1), repeats_as_pshape);
    std::vector<Dimension> repeats_value(repeats_as_pshape);
    if (repeats_are_known && !repeats_value.empty() && arg_shape.rank().is_static())
    {
        std::vector<Dimension> data_shape(arg_shape);
        auto data_rank = data_shape.size();
        auto repeats_rank = repeats_value.size();
        auto output_rank = std::max(data_rank, repeats_rank);

        // expand data shape and repeats to output rank
        data_shape.insert(data_shape.begin(), output_rank - data_rank, 1);
        repeats_value.insert(repeats_value.begin(), output_rank - repeats_rank, 1);

        auto output_shape = PartialShape::dynamic(output_rank);
        for (size_t i = 0; i < output_rank; i++)
            output_shape[i] = data_shape[i] * repeats_value[i];
        set_output_type(0, arg_et, output_shape);
    }
    else
    {
        set_output_type(0, arg_et, PartialShape::dynamic());
    }

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
}

shared_ptr<Node> op::v0::Tile::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Tile_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Tile>(new_args.at(0), new_args.at(1));
}

bool op::v0::Tile::evaluate_tile(const HostTensorVector& outputs,
                                 const HostTensorVector& inputs) const
{
    const auto& data = inputs[0];
    const auto& axis = inputs[1];
    auto& output = outputs[0];
    auto repeats_val = read_vector<int64_t>(axis);
    auto repeats_rank = repeats_val.size();
    Shape data_shape = data->get_shape();
    auto data_rank = data_shape.size();
    auto output_rank = std::max(data_rank, repeats_rank);

    // expand data shape and repeats to output rank
    data_shape.insert(data_shape.begin(), output_rank - data_rank, 1);
    repeats_val.insert(repeats_val.begin(), output_rank - repeats_rank, 1);

    Shape output_shape(output_rank);
    for (size_t i = 0; i < output_rank; i++)
    {
        output_shape[i] = data_shape[i] * repeats_val[i];
    }

    if (!output->get_is_allocated())
    {
        output->set_shape(output_shape);
    }

    runtime::reference::tile(data->get_data_ptr<const char>(),
                             output->get_data_ptr<char>(),
                             data->get_shape(),
                             output_shape,
                             data->get_element_type().size(),
                             repeats_val);

    return true;
}

bool op::v0::Tile::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Tile_evaluate);
    return evaluate_tile(outputs, inputs);
}

bool op::v0::Tile::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v0_Tile_has_evaluate);
    return true;
}
