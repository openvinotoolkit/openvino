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
#include <ngraph/op/constant.hpp>
#include <ngraph/ops.hpp>
#include <numeric>

#include "depth_to_space.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"

#include "ngraph/runtime/reference/depth_to_space.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

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
    visitor.on_attribute("block_size", m_blocksize);
    visitor.on_attribute("mode", m_mode);
    return true;
}

shared_ptr<Node> op::DepthToSpace::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<DepthToSpace>(new_args.at(0), m_mode, m_blocksize);
}

void op::DepthToSpace::validate_and_infer_types()
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
        set_output_type(0, data_type, PartialShape::dynamic());
    }
}

bool op::DepthToSpace::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
    if (inputs[0]->get_partial_shape().is_dynamic())
    {
        return false;
    }
    runtime::reference::depth_to_space(inputs[0]->get_data_ptr<const char>(),
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
