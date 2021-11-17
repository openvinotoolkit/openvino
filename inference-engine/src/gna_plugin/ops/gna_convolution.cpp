// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softsign.hpp"

#include <ngraph/validation_util.hpp>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/runtime/host_tensor.hpp"

#include <cmath>
#include <cstddef>

NGRAPH_RTTI_DEFINITION(GNAPluginNS::GNAConvolution, "GNAConvolution", 0);

namespace GNAPluginNS {

GNAConvolution::GNAConvolution(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Output<Node>& bias,
                                 const Strides& strides,
                                 const CoordinateDiff& pads_begin,
                                 const CoordinateDiff& pads_end,
                                 const Strides& dilations,
                                 const PadType& auto_pad)
    : Op({data_batch, filters, bias}),
      m_strides(strides),
      m_dilations(dilations),
      m_pads_begin(pads_begin),
      m_pads_end(pads_end),
      m_auto_pad(auto_pad) {
    constructor_validate_and_infer_types();
}

bool GNAConvolution::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void GNAConvolution::validate_and_infer_types() {
    element::Type data_batch_et = get_input_element_type(0);
    element::Type filters_et = get_input_element_type(1);

    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, data_batch_et, filters_et),
                          "Element types for data batch and filters do not match (data batch element type: ",
                          data_batch_et,
                          ", filters element type: ",
                          filters_et,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          result_et.is_real() || result_et.is_integral_number(),
                          "Element types must be numeric. Got: ",
                          result_et);
    auto& data_shape = get_input_partial_shape(0);
    auto& filter_shape = get_input_partial_shape(1);

    m_num_spatial = calculate_num_spatial(this, data_shape, filter_shape, 2, 2);
    update_and_validate_attributes(this);

    std::vector<ov::PartialShape> input_shapes = {data_shape, filter_shape};
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};

    if (m_num_spatial != -1) {
        resolve_auto_pad_for_shape(this, m_pads_begin, m_pads_end, input_shapes, 2, 2);
        shape_infer(this, m_pads_begin, m_pads_end, input_shapes, output_shapes);
    }

    set_output_type(0, result_et, output_shapes[0]);
}

shared_ptr<Node> GNAConvolution::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<v1::Convolution>(new_args.at(0),
                                        new_args.at(1),
                                        new_args.at(2),
                                        m_strides,
                                        m_pads_begin,
                                        m_pads_end,
                                        m_dilations,
                                        m_auto_pad);
}

} // namespace GNAPluginNS
