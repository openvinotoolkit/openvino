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
#include <numeric>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/space_to_depth.hpp"
#include "ngraph/shape.hpp"

#include "ngraph/runtime/reference/space_to_depth.hpp"

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
    visitor.on_attribute("block_size", m_blocksize);
    visitor.on_attribute("mode", m_mode);
    return true;
}

shared_ptr<Node> op::SpaceToDepth::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<SpaceToDepth>(new_args.at(0), m_mode, m_blocksize);
}

void ngraph::op::v0::SpaceToDepth::validate_and_infer_types()
{
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
        set_output_type(0, data_type, PartialShape::dynamic());
    }
}

bool ngraph::op::v0::SpaceToDepth::evaluate(const HostTensorVector& outputs,
                                            const HostTensorVector& inputs) const
{
    if (inputs[0]->get_partial_shape().is_dynamic())
    {
        return false;
    }
    runtime::reference::space_to_depth(inputs[0]->get_data_ptr<const char>(),
                                       outputs[0]->get_data_ptr<char>(),
                                       inputs[0]->get_shape(),
                                       outputs[0]->get_shape(),
                                       m_blocksize,
                                       m_mode,
                                       inputs[0]->get_element_type().size());
    return true;
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
