// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/space_to_depth.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <space_to_depth_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/runtime/reference/space_to_depth.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ov::op::v0::SpaceToDepth);

ov::op::v0::SpaceToDepth::SpaceToDepth(const Output<Node>& data, const SpaceToDepthMode& mode, size_t block_size)
    : Op({data}),
      m_blocksize(block_size),
      m_mode(mode) {
    constructor_validate_and_infer_types();
}

ov::op::v0::SpaceToDepth::SpaceToDepth(const Output<Node>& data, const std::string& mode, size_t block_size)
    : SpaceToDepth(data, as_enum<SpaceToDepthMode>(mode), block_size) {}

bool ngraph::op::v0::SpaceToDepth::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_SpaceToDepth_visit_attributes);
    visitor.on_attribute("block_size", m_blocksize);
    visitor.on_attribute("mode", m_mode);
    return true;
}

std::shared_ptr<Node> ov::op::v0::SpaceToDepth::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_SpaceToDepth_clone_with_new_inputs);
    if (new_args.size() != 1) {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<SpaceToDepth>(new_args.at(0), m_mode, m_blocksize);
}

void ngraph::op::v0::SpaceToDepth::validate_and_infer_types() {
    OV_OP_SCOPE(v0_SpaceToDepth_validate_and_infer_types);

    const auto& data_type = get_input_element_type(0);
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    const std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0)};
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, data_type, output_shapes[0]);
}

namespace {
bool evaluate_space_to_depth(const HostTensorVector& outputs,
                             const HostTensorVector& inputs,
                             const std::size_t block_size,
                             const ov::op::v0::SpaceToDepth::SpaceToDepthMode mode) {
    const auto& in = inputs[0];
    const auto& out = outputs[0];
    size_t elem_size = in->get_element_type().size();

    if (in->get_partial_shape().is_dynamic()) {
        return false;
    }

    runtime::reference::space_to_depth(in->get_data_ptr<char>(),
                                       in->get_shape(),
                                       out->get_data_ptr<char>(),
                                       out->get_shape(),
                                       block_size,
                                       mode,
                                       elem_size);
    return true;
}
}  // namespace

bool ngraph::op::v0::SpaceToDepth::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_SpaceToDepth_evaluate);
    return evaluate_space_to_depth(outputs, inputs, m_blocksize, m_mode);
}

bool ngraph::op::v0::SpaceToDepth::has_evaluate() const {
    OV_OP_SCOPE(v0_SpaceToDepth_has_evaluate);
    return !get_input_partial_shape(0).is_dynamic();
}

std::ostream& ov::operator<<(std::ostream& s, const op::v0::SpaceToDepth::SpaceToDepthMode& type) {
    return s << as_string(type);
}

namespace ov {
template <>
NGRAPH_API EnumNames<ngraph::op::v0::SpaceToDepth::SpaceToDepthMode>&
EnumNames<ngraph::op::v0::SpaceToDepth::SpaceToDepthMode>::get() {
    static auto enum_names = EnumNames<ngraph::op::v0::SpaceToDepth::SpaceToDepthMode>(
        "op::v0::SpaceToDepth::SpaceToDepthMode",
        {{"blocks_first", ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST},
         {"depth_first", ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST}});
    return enum_names;
}

BWDCMP_RTTI_DEFINITION(AttributeAdapter<op::v0::SpaceToDepth::SpaceToDepthMode>);
}  // namespace ov
