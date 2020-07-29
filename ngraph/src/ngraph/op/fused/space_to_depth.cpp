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

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/shape.hpp"
#include "space_to_depth.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::SpaceToDepth::type_info;

op::SpaceToDepth::SpaceToDepth(const Output<Node>& data,
                               const SpaceToDepthMode& mode,
                               size_t block_size)
    : FusedOp({data})
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
    visitor.on_attribute("block_size", m_blocksize);
    visitor.on_attribute("mode", m_mode);
    return true;
}

OutputVector op::SpaceToDepth::decompose_op() const
{
    auto data = input_value(0);
    auto data_shape = data.get_shape();

    NODE_VALIDATION_CHECK(this,
                          (data_shape.size() >= 3),
                          "The input tensor with rank lower than 3 is not supported (input rank: ",
                          data_shape.size(),
                          ")");

    NODE_VALIDATION_CHECK(this, m_blocksize > 0, "m_blocksize must be greater than 0");

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
    auto flat_node = builder::opset1::reshape(data, dispersed_shape);
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
    // y = reshape(x'', [N, C * (block_size ^ K), D1 / block_size, D2 / block_size, ..., DK /
    // block_size])
    case SpaceToDepthMode::DEPTH_FIRST:
    {
        axes_order.insert(axes_order.begin() + 1, 1);
        break;
    }
    // x' = reshape(data, [N, C, D1/block_size, block_size, D2/block_size, block_size, ... ,
    // DK/block_size, block_size])
    // x'' = transpose(x',  [0,  3, 5, ..., K + (K + 1), 1,  2, 4, ..., K + K])
    // y = reshape(x'', [N, C * (block_size ^ K), D1 / block_size, D2 / block_size, ..., DK /
    // block_size])
    case SpaceToDepthMode::BLOCKS_FIRST:
    default: { axes_order.insert(axes_order.begin() + spatial_dims + 1, 1);
    }
    }
    flat_node = builder::opset1::reorder_axes(flat_node, axes_order);
    Shape squeezed_shape{n_dim};
    for (int i = 0; i < spatial_dims; ++i)
    {
        squeezed_shape.push_back(data_shape.at(spatial_dim_index + i) / m_blocksize);
    }
    squeezed_shape.insert(squeezed_shape.begin() + 1, c_dim * std::pow(m_blocksize, spatial_dims));
    flat_node = builder::opset1::reshape(flat_node, squeezed_shape);

    return OutputVector{flat_node};
}

shared_ptr<Node> op::SpaceToDepth::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<SpaceToDepth>(new_args.at(0), m_mode, m_blocksize);
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
