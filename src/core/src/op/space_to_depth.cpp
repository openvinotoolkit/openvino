// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/space_to_depth.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>

#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/reference/space_to_depth.hpp"
#include "space_to_depth_shape_inference.hpp"

namespace ov {
namespace op {
namespace v0 {
SpaceToDepth::SpaceToDepth(const Output<Node>& data, const SpaceToDepthMode& mode, size_t block_size)
    : Op({data}),
      m_blocksize(block_size),
      m_mode(mode) {
    constructor_validate_and_infer_types();
}

SpaceToDepth::SpaceToDepth(const Output<Node>& data, const std::string& mode, size_t block_size)
    : SpaceToDepth(data, as_enum<SpaceToDepthMode>(mode), block_size) {}

bool SpaceToDepth::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_SpaceToDepth_visit_attributes);
    visitor.on_attribute("block_size", m_blocksize);
    visitor.on_attribute("mode", m_mode);
    return true;
}

std::shared_ptr<Node> SpaceToDepth::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_SpaceToDepth_clone_with_new_inputs);
    if (new_args.size() != 1) {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
    return std::make_shared<SpaceToDepth>(new_args.at(0), m_mode, m_blocksize);
}

void SpaceToDepth::validate_and_infer_types() {
    OV_OP_SCOPE(v0_SpaceToDepth_validate_and_infer_types);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool SpaceToDepth::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_SpaceToDepth_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto& in = inputs[0];
    const auto& out = outputs[0];
    reference::space_to_depth(static_cast<const char*>(in.data()),
                              in.get_shape(),
                              static_cast<char*>(out.data()),
                              out.get_shape(),
                              m_blocksize,
                              m_mode,
                              in.get_element_type().size());
    return true;
}

bool SpaceToDepth::has_evaluate() const {
    OV_OP_SCOPE(v0_SpaceToDepth_has_evaluate);
    return !get_input_partial_shape(0).is_dynamic();
}

void SpaceToDepth::set_block_size(size_t block_size) {
    m_blocksize = block_size;
}

void SpaceToDepth::set_mode(SpaceToDepthMode mode) {
    m_mode = mode;
}
}  // namespace v0
}  // namespace op

std::ostream& operator<<(std::ostream& s, const op::v0::SpaceToDepth::SpaceToDepthMode& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<op::v0::SpaceToDepth::SpaceToDepthMode>&
EnumNames<op::v0::SpaceToDepth::SpaceToDepthMode>::get() {
    static auto enum_names = EnumNames<op::v0::SpaceToDepth::SpaceToDepthMode>(
        "op::v0::SpaceToDepth::SpaceToDepthMode",
        {{"blocks_first", op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST},
         {"depth_first", op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST}});
    return enum_names;
}
}  // namespace ov
