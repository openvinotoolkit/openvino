// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/gather_compressed.hpp"
#include "gather_shape_inference.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

GatherCompressed::GatherCompressed(const ov::Output<Node>& data,
                                   const ov::Output<Node>& indices,
                                   const ov::Output<Node>& axis,
                                   const ov::Output<Node>& decompression_scale,
                                   const ov::Output<Node>& decompression_zero_point,
                                   const ov::element::Type output_type)
    : ov::op::v8::Gather({data, indices, axis}), m_output_type(output_type) {
    set_argument(3, decompression_scale);
    set_argument(4, decompression_zero_point);
    validate_and_infer_types();
}

GatherCompressed::GatherCompressed(const ov::Output<Node>& data,
                                   const ov::Output<Node>& indices,
                                   const ov::Output<Node>& axis,
                                   const ov::Output<Node>& decompression_scale,
                                   const ov::element::Type output_type)
    : ov::op::v8::Gather({data, indices, axis}), m_output_type(output_type) {
    set_argument(3, decompression_scale);
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> GatherCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    if (new_args.size() == 4)
        return std::make_shared<GatherCompressed>(new_args.at(0),
                                                  new_args.at(1),
                                                  new_args.at(2),
                                                  new_args.at(3),
                                                  m_output_type);
    else if (new_args.size() == 5)
        return std::make_shared<GatherCompressed>(new_args.at(0),
                                                  new_args.at(1),
                                                  new_args.at(2),
                                                  new_args.at(3),
                                                  new_args.at(4),
                                                  m_output_type);
    else
        OPENVINO_THROW("Unexpected inputs count for GatherCompressed op: ", new_args.size());
}

void GatherCompressed::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size >= 3,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected at least 3.");

    auto out_shapes = ov::op::shape_infer(this, std::vector<ov::PartialShape>{get_input_partial_shape(0),
                                                get_input_partial_shape(1), get_input_partial_shape(2)});

    auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, out_shapes[0]);
}

bool GatherCompressed::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
