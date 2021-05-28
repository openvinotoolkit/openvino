// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstddef>
#include <memory>
#include <ngraph/op/constant.hpp>
#include <ngraph/ops.hpp>
#include <numeric>
#include "itt.hpp"

#include "ngraph/builder/reshape.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/depth_to_space.hpp"
#include "ngraph/shape.hpp"

#include "ngraph/runtime/opt_kernel/reshape.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v0::DepthToSpace, "DepthToSpace", 0);

op::DepthToSpace::DepthToSpace(const Output<Node>& data,
                               const DepthToSpaceMode& mode,
                               const size_t block_size)
    : Op({data})
    , m_blocksize(block_size)
    , m_mode(mode)
{
    constructor_validate_and_infer_types();
}

op::DepthToSpace::DepthToSpace(const Output<Node>& data,
                               const std::string& mode,
                               const size_t block_size)
    : DepthToSpace(data, mode_from_string(mode), block_size)
{
}

bool op::DepthToSpace::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_DepthToSpace_visit_attributes);
    visitor.on_attribute("block_size", m_blocksize);
    visitor.on_attribute("mode", m_mode);
    return true;
}

shared_ptr<Node> op::DepthToSpace::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_DepthToSpace_clone_with_new_inputs);
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<DepthToSpace>(new_args.at(0), m_mode, m_blocksize);
}

void op::DepthToSpace::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_DepthToSpace_validate_and_infer_types);
    PartialShape data_pshape = get_input_partial_shape(0);

    const auto& data_type = get_input_element_type(0);

    auto data = input_value(0);

    if (data_pshape.is_static())
    {
        const auto& data_shape = data.get_shape();

        NODE_VALIDATION_CHECK(
            this,
            !(data_shape.size() < 3),
            "The input tensor with rank lower than 3 is not supported (input rank: ",
            data_shape.size(),
            ")");

        auto divider = std::pow(m_blocksize, data_shape.size() - 2);
        NODE_VALIDATION_CHECK(this, (divider), "DepthToSpace: The divider must not be 0");

        NODE_VALIDATION_CHECK(this,
                              m_blocksize > 0 && !(data_shape[1] % m_blocksize),
                              "DepthToSpace: The input data's 'channels' axis size: ",
                              data_shape[1],
                              " must be a equivalent to 'block_size'^'spatial_dims': ",
                              divider);

        auto out_shape = data_shape;
        out_shape[1] /= divider;
        for (size_t i = 2; i < out_shape.size(); i++)
        {
            out_shape[i] *= m_blocksize;
        }

        set_output_size(1);
        set_output_type(0, data_type, out_shape);
    }
    else
    {
        set_output_type(0, data_type, PartialShape::dynamic(data_pshape.rank()));
    }
}

bool op::DepthToSpace::evaluate_depth_to_space(const HostTensorVector& outputs,
                                               const HostTensorVector& inputs) const
{
    const auto& data = inputs[0];
    const auto& out = outputs[0];
    size_t elem_size = data->get_element_type().size();

    if (data->get_partial_shape().is_dynamic())
    {
        return false;
    }
    auto data_shape = data->get_shape();
    const size_t n_dim = data_shape.at(0);
    const size_t c_dim = data_shape.at(1);
    const size_t spatial_dim_index = 2;
    const size_t spatial_dims = data_shape.size() - spatial_dim_index;
    const auto c_dim_divider = static_cast<int>(std::pow(m_blocksize, spatial_dims));

    NODE_VALIDATION_CHECK(this,
                          m_blocksize > 0 && c_dim % c_dim_divider == 0,
                          "DepthToSpace: The input data's 'channels' axis size: ",
                          c_dim,
                          " must be a equivalent to ",
                          "'block_size'^'spatial_dims': ",
                          c_dim_divider);

    auto bs = static_cast<size_t>(m_blocksize);
    size_t c_flat = c_dim / c_dim_divider;

    // First we have to disperse the data from depth channel, then rearrange them
    // so as appropriate chunks of data where close to their destination place.
    // Finally squeeze data from respective dimensions.
    shared_ptr<Node> flat_node;
    Shape dispersed_shape{n_dim};
    for (size_t i = 0; i < spatial_dims; ++i)
    {
        dispersed_shape.push_back(bs);
    }
    for (size_t i = 0; i < spatial_dims; ++i)
    {
        dispersed_shape.push_back(data_shape.at(spatial_dim_index + i));
    }
    vector<size_t> axes_order{0};
    switch (m_mode)
    {
    // x' = reshape(data, [N, C / (block_size ^ K), block_size, block_size, ..., block_size,
    // D1, D2,
    // ..., DK])
    // x'' = transpose(x', [0,  1,  K + 2, 2, K + 3, 3, K + 4, 4, ..., K + (K + 1), K + 1])
    // y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 * block_size, D3 *
    // block_size,
    // ..., DK * block_size])
    case DepthToSpaceMode::DEPTH_FIRST:
    {
        dispersed_shape.insert(dispersed_shape.begin() + 1, c_flat);
        axes_order.push_back(1);
        for (size_t i = spatial_dim_index; i < data_shape.size(); ++i)
        {
            axes_order.push_back(spatial_dims + i);
            axes_order.push_back(i);
        }

        break;
    }
    // x' = reshape(data, [N, block_size, block_size, ..., block_size, C / (block_size ^ K),
    // D1, D2,
    // ..., DK])
    // x'' = transpose(x', [0,  K + 1,  K + 2, 1, K + 3, 2, K + 4, 3, ..., K + (K + 1), K])
    // y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 * block_size, D3 *
    // block_size,
    // ..., DK * block_size])
    case DepthToSpaceMode::BLOCKS_FIRST:
    default:
    {
        dispersed_shape.insert(dispersed_shape.begin() + spatial_dims + 1, c_flat);
        axes_order.push_back(spatial_dims + 1);
        for (size_t i = 2; i < data_shape.size(); ++i)
        {
            axes_order.push_back(spatial_dims + i);
            axes_order.push_back(i - 1);
        }
        break;
    }
    }
    std::vector<size_t> plain_axes_order(data_shape.size());
    std::iota(plain_axes_order.begin(), plain_axes_order.end(), 0);
    std::vector<char> dispersed_data(shape_size(data_shape) * elem_size);
    std::vector<char> transposed_data(shape_size(data_shape) * elem_size);

    runtime::opt_kernel::reshape(data->get_data_ptr<char>(),
                                 dispersed_data.data(),
                                 data_shape,
                                 plain_axes_order,
                                 dispersed_shape,
                                 elem_size);

    Shape post_transpose_shape(axes_order.size());
    for (size_t axis_idx = 0; axis_idx < axes_order.size(); ++axis_idx)
    {
        post_transpose_shape[axis_idx] = dispersed_shape[axes_order[axis_idx]];
    }
    runtime::opt_kernel::reshape(dispersed_data.data(),
                                 transposed_data.data(),
                                 dispersed_shape,
                                 axes_order,
                                 post_transpose_shape,
                                 elem_size);

    Shape squeezed_shape{n_dim, c_flat};
    for (size_t i = spatial_dim_index; i < data_shape.size(); ++i)
    {
        squeezed_shape.push_back(data_shape.at(i) * bs);
    }
    for (size_t i = plain_axes_order.size() - 1; i < post_transpose_shape.size() - 1; ++i)
    {
        plain_axes_order.push_back(plain_axes_order[i] + 1);
    }
    runtime::opt_kernel::reshape(transposed_data.data(),
                                 out->get_data_ptr<char>(),
                                 post_transpose_shape,
                                 plain_axes_order,
                                 squeezed_shape,
                                 elem_size);
    return true;
}

bool op::DepthToSpace::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_DepthToSpace_evaluate);
    return evaluate_depth_to_space(outputs, inputs);
}

bool op::DepthToSpace::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v0_DepthToSpace_has_evaluate);
    return !get_input_partial_shape(0).is_dynamic();
}

namespace ngraph
{
    template <>
    NGRAPH_API EnumNames<op::DepthToSpace::DepthToSpaceMode>&
        EnumNames<op::DepthToSpace::DepthToSpaceMode>::get()
    {
        static auto enum_names = EnumNames<op::DepthToSpace::DepthToSpaceMode>(
            "op::DepthToSpace::DepthToSpaceMode",
            {{"blocks_first", op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST},
             {"depth_first", op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::DepthToSpace::DepthToSpaceMode>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::DepthToSpace::DepthToSpaceMode& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph

op::DepthToSpace::DepthToSpaceMode op::DepthToSpace::mode_from_string(const std::string& mode) const
{
    return as_enum<DepthToSpaceMode>(mode);
}
