//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/space_to_depth.hpp"
#include "ngraph/shape.hpp"

#include "ngraph/runtime/opt_kernel/reshape.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

constexpr NodeTypeInfo op::SpaceToDepth::type_info;

op::SpaceToDepth::SpaceToDepth(const Output<Node>& data,
                               const SpaceToDepthMode& mode,
                               size_t block_size)
    : Op({data})
    , m_blocksize(block_size)
    , m_mode(mode)
{
    constructor_validate_and_infer_types();
}

op::SpaceToDepth::SpaceToDepth(const Output<Node>& data, const std::string& mode, size_t block_size)
    : SpaceToDepth(data, as_enum<SpaceToDepthMode>(mode), block_size)
{
}

bool ngraph::op::v0::SpaceToDepth::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_SpaceToDepth_visit_attributes);
    visitor.on_attribute("block_size", m_blocksize);
    visitor.on_attribute("mode", m_mode);
    return true;
}

shared_ptr<Node> op::SpaceToDepth::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_SpaceToDepth_clone_with_new_inputs);
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<SpaceToDepth>(new_args.at(0), m_mode, m_blocksize);
}

void ngraph::op::v0::SpaceToDepth::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_SpaceToDepth_validate_and_infer_types);
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

        auto multiplier = std::pow(m_blocksize, data_shape.size() - 2);

        auto out_shape = data_shape;
        out_shape[1] *= multiplier;
        for (size_t i = 2; i < out_shape.size(); i++)
        {
            NODE_VALIDATION_CHECK(this,
                                  m_blocksize > 0 && !(out_shape[i] % m_blocksize),
                                  "The dimension on position: ",
                                  i,
                                  " equal to: ",
                                  out_shape[i],
                                  " must be a multiple of m_blocksize: ",
                                  m_blocksize);

            out_shape[i] /= m_blocksize;
        }

        set_output_size(1);
        set_output_type(0, data_type, out_shape);
    }
    else
    {
        set_output_type(0, data_type, PartialShape::dynamic(data_pshape.rank()));
    }
}

bool ngraph::op::v0::SpaceToDepth::evaluate_space_to_depth(const HostTensorVector& outputs,
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

    for (int i = spatial_dim_index; i < data_shape.size(); ++i)
    {
        NODE_VALIDATION_CHECK(this,
                              m_blocksize > 0 && data_shape.at(i) % m_blocksize == 0,
                              "The dimension on position: ",
                              i,
                              " equal to: ",
                              data_shape.at(i),
                              " must be a multiple of m_blocksize: ",
                              m_blocksize);
    }

    // First we have to disperse the data from spatial dimensions, then
    // rearrange them so as appropriate chunks of data where close to their
    // destination place. Finally squeeze data from respective dimensions.
    Shape dispersed_shape{n_dim, c_dim};
    for (int i = 0; i < spatial_dims; ++i)
    {
        dispersed_shape.push_back(data_shape.at(i + spatial_dim_index) / m_blocksize);
        dispersed_shape.push_back(m_blocksize);
    }
    std::vector<size_t> plain_axes_order(data_shape.size());
    std::iota(plain_axes_order.begin(), plain_axes_order.end(), 0);
    std::vector<char> dispersed_data(shape_size(data_shape) * elem_size);
    runtime::opt_kernel::reshape(data->get_data_ptr<char>(),
                                 dispersed_data.data(),
                                 data_shape,
                                 plain_axes_order,
                                 dispersed_shape,
                                 elem_size);
    // calculate axes to transpose
    // [0, 3, 5, ..., spatial_dims + (spatial_dims + 1), 2, 4, ..., K + K])
    vector<size_t> axes_order{0};
    for (size_t i = 0, j = 3; i < spatial_dims; ++i, j += 2)
    {
        axes_order.push_back(j);
    }
    for (size_t i = 0, j = 2; i < spatial_dims; ++i, j += 2)
    {
        axes_order.push_back(j);
    }

    switch (m_mode)
    {
    // x' = reshape(data, [N, C, D1/block_size, block_size, D2/block_size, block_size, ...,
    // DK/block_size, block_size])
    // x'' = transpose(x', [0,  1, 3, 5, ..., K + (K + 1),  2, 4, ..., K + K])
    // y = reshape(x'', [N, C * (block_size ^ K), D1 / block_size, D2 / block_size, ..., DK
    // /
    // block_size])
    case SpaceToDepthMode::DEPTH_FIRST:
    {
        axes_order.insert(axes_order.begin() + 1, 1);
        break;
    }
    // x' = reshape(data, [N, C, D1/block_size, block_size, D2/block_size, block_size, ... ,
    // DK/block_size, block_size])
    // x'' = transpose(x',  [0,  3, 5, ..., K + (K + 1), 1,  2, 4, ..., K + K])
    // y = reshape(x'', [N, C * (block_size ^ K), D1 / block_size, D2 / block_size, ..., DK
    // /
    // block_size])
    case SpaceToDepthMode::BLOCKS_FIRST:
    default:
    {
        axes_order.insert(axes_order.begin() + spatial_dims + 1, 1);
    }
    }
    std::vector<char> transposed_data(shape_size(data_shape) * elem_size);
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

    Shape squeezed_shape{n_dim};
    for (int i = 0; i < spatial_dims; ++i)
    {
        squeezed_shape.push_back(data_shape.at(spatial_dim_index + i) / m_blocksize);
    }
    squeezed_shape.insert(squeezed_shape.begin() + 1, c_dim * std::pow(m_blocksize, spatial_dims));
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
bool ngraph::op::v0::SpaceToDepth::evaluate(const HostTensorVector& outputs,
                                            const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_SpaceToDepth_evaluate);
    return evaluate_space_to_depth(outputs, inputs);
}

namespace ngraph
{
    template <>
    EnumNames<op::v0::SpaceToDepth::SpaceToDepthMode>&
        EnumNames<op::v0::SpaceToDepth::SpaceToDepthMode>::get()
    {
        static auto enum_names = EnumNames<op::v0::SpaceToDepth::SpaceToDepthMode>(
            "op::v0::SpaceToDepth::SpaceToDepthMode",
            {{"blocks_first", op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST},
             {"depth_first", op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v0::SpaceToDepth::SpaceToDepthMode>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::v0::SpaceToDepth::SpaceToDepthMode& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph
