// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/depth_to_space.hpp"

#include <cmath>
#include <cstddef>
#include <memory>

#include "depth_to_space_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/reference/depth_to_space.hpp"

namespace ov {
namespace op {
namespace v0 {
DepthToSpace::DepthToSpace(const Output<Node>& data, const DepthToSpaceMode& mode, const size_t block_size)
    : Op({data}),
      m_blocksize(block_size),
      m_mode(mode) {
    constructor_validate_and_infer_types();
}

DepthToSpace::DepthToSpace(const Output<Node>& data, const std::string& mode, const size_t block_size)
    : DepthToSpace(data, as_enum<DepthToSpaceMode>(mode), block_size) {}

bool DepthToSpace::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_DepthToSpace_visit_attributes);
    visitor.on_attribute("block_size", m_blocksize);
    visitor.on_attribute("mode", m_mode);
    return true;
}

std::shared_ptr<Node> DepthToSpace::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_DepthToSpace_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<DepthToSpace>(new_args.at(0), m_mode, m_blocksize);
}

void DepthToSpace::validate_and_infer_types() {
    OV_OP_SCOPE(v0_DepthToSpace_validate_and_infer_types);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool DepthToSpace::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_DepthToSpace_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto& in = inputs[0];
    const auto& out = outputs[0];
    reference::depth_to_space(static_cast<const char*>(in.data()),
                              in.get_shape(),
                              static_cast<char*>(out.data()),
                              out.get_shape(),
                              m_blocksize,
                              m_mode,
                              in.get_element_type().size());
    return true;
}

bool DepthToSpace::has_evaluate() const {
    OV_OP_SCOPE(v0_DepthToSpace_has_evaluate);
    return !get_input_partial_shape(0).is_dynamic();
}

void DepthToSpace::set_block_size(size_t block_size) {
    m_blocksize = block_size;
}

void DepthToSpace::set_mode(DepthToSpaceMode mode) {
    m_mode = mode;
}
}  // namespace v0
}  // namespace op

std::ostream& operator<<(std::ostream& s, const op::v0::DepthToSpace::DepthToSpaceMode& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<op::v0::DepthToSpace::DepthToSpaceMode>&
EnumNames<op::v0::DepthToSpace::DepthToSpaceMode>::get() {
    static auto enum_names = EnumNames<op::v0::DepthToSpace::DepthToSpaceMode>(
        "op::DepthToSpace::DepthToSpaceMode",
        {{"blocks_first", op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST},
         {"depth_first", op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST}});
    return enum_names;
}
}  // namespace ov
