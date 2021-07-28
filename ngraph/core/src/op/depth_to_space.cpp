// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/depth_to_space.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <ngraph/op/constant.hpp>
#include <ngraph/ops.hpp>
#include "itt.hpp"

#include "ngraph/runtime/reference/depth_to_space.hpp"
#include "ngraph/shape.hpp"

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
    : DepthToSpace(data, as_enum<DepthToSpaceMode>(mode), block_size)
{
}

bool op::DepthToSpace::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_DepthToSpace_visit_attributes);
    visitor.on_attribute("block_size", m_blocksize);
    visitor.on_attribute("mode", m_mode);
    return true;
}

std::shared_ptr<Node> op::DepthToSpace::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_DepthToSpace_clone_with_new_inputs);
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<DepthToSpace>(new_args.at(0), m_mode, m_blocksize);
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

bool evaluate_depth_to_space(const HostTensorVector& outputs,
                             const HostTensorVector& inputs,
                             const std::size_t block_size,
                             const op::DepthToSpace::DepthToSpaceMode mode)
{
    const auto& in = inputs[0];
    const auto& out = outputs[0];
    const size_t elem_size = in->get_element_type().size();
    if (in->get_partial_shape().is_dynamic())
    {
        return false;
    }
    runtime::reference::depth_to_space(in->get_data_ptr<char>(),
                                       in->get_shape(),
                                       out->get_data_ptr<char>(),
                                       out->get_shape(),
                                       block_size,
                                       mode,
                                       elem_size);
    return true;
}

bool op::DepthToSpace::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_DepthToSpace_evaluate);
    return evaluate_depth_to_space(outputs, inputs, m_blocksize, m_mode);
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
