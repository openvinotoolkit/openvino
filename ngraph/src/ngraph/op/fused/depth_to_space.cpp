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
#include <cmath>
#include <cstddef>
#include <memory>

#include "depth_to_space.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::DepthToSpace::type_info;

op::DepthToSpace::DepthToSpace(const Output<Node>& data,
                               const DepthToSpaceMode& mode,
                               const size_t block_size)
    : FusedOp({data})
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
    visitor.on_attribute("block_size", m_blocksize);
    visitor.on_attribute("mode", m_mode);
    return true;
}

NodeVector op::DepthToSpace::decompose_op() const
{
    auto data = input_value(0);
    auto data_shape = data.get_shape();

    NODE_VALIDATION_CHECK(this,
                          (data_shape.size() >= 3),
                          "The input tensor with rank lower than 3 is not supported (input rank: ",
                          data_shape.size(),
                          ")");

    if (data_shape.size() == 3)
    {
        // Insert batch axis
        data_shape.insert(data_shape.begin(), 1);
        data = builder::opset1::reshape(data, data_shape);
    }
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
    for (int i = 0; i < spatial_dims; ++i)
    {
        dispersed_shape.push_back(bs);
    }
    for (int i = 0; i < spatial_dims; ++i)
    {
        dispersed_shape.push_back(data_shape.at(spatial_dim_index + i));
    }
    vector<size_t> axes_order{0};
    switch (m_mode)
    {
    // x' = reshape(data, [N, C / (block_size ^ K), block_size, block_size, ..., block_size, D1, D2,
    // ..., DK])
    // x'' = transpose(x', [0,  1,  K + 2, 2, K + 3, 3, K + 4, 4, ..., K + (K + 1), K + 1])
    // y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 * block_size, D3 * block_size,
    // ..., DK * block_size])
    case DepthToSpaceMode::DEPTH_FIRST:
    {
        dispersed_shape.insert(dispersed_shape.begin() + 1, c_flat);
        flat_node = builder::opset1::reshape(data, dispersed_shape);

        axes_order.push_back(1);
        for (int i = spatial_dim_index; i < data_shape.size(); ++i)
        {
            axes_order.push_back(spatial_dims + i);
            axes_order.push_back(i);
        }

        flat_node = builder::opset1::reorder_axes(flat_node, axes_order);
        break;
    }
    // x' = reshape(data, [N, block_size, block_size, ..., block_size, C / (block_size ^ K), D1, D2,
    // ..., DK])
    // x'' = transpose(x', [0,  K + 1,  K + 2, 1, K + 3, 2, K + 4, 3, ..., K + (K + 1), K])
    // y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 * block_size, D3 * block_size,
    // ..., DK * block_size])
    case DepthToSpaceMode::BLOCKS_FIRST:
    default:
    {
        dispersed_shape.insert(dispersed_shape.begin() + spatial_dims + 1, c_flat);
        flat_node = builder::opset1::reshape(data, dispersed_shape);

        axes_order.push_back(spatial_dims + 1);
        for (int i = 2; i < data_shape.size(); ++i)
        {
            axes_order.push_back(spatial_dims + i);
            axes_order.push_back(i - 1);
        }
        flat_node = builder::opset1::reorder_axes(flat_node, axes_order);
    }
    }
    Shape squeezed_shape{n_dim, c_flat};
    for (int i = spatial_dim_index; i < data_shape.size(); ++i)
    {
        squeezed_shape.push_back(data_shape.at(i) * bs);
    }
    flat_node = builder::opset1::reshape(flat_node, squeezed_shape);

    return NodeVector{flat_node};
}

shared_ptr<Node> op::DepthToSpace::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<DepthToSpace>(new_args.at(0), m_mode, m_blocksize);
}

namespace ngraph
{
    template <>
    EnumNames<op::DepthToSpace::DepthToSpaceMode>&
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
}

op::DepthToSpace::DepthToSpaceMode op::DepthToSpace::mode_from_string(const std::string& mode) const
{
    return as_enum<DepthToSpaceMode>(mode);
}
