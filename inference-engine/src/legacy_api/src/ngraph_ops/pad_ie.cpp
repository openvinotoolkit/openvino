// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/pad_ie.hpp"

#include <assert.h>

#include <memory>
#include <string>
#include <transformations/utils/utils.hpp>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/pad.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::PadIE::type_info;

op::PadIE::PadIE(const std::shared_ptr<op::v1::Pad>& pad)
    : Op({pad->input(0).get_source_output()}),
      m_pad_mode(pad->get_pad_mode()),
      m_pads_begin(pad->get_pads_begin()),
      m_pads_end(pad->get_pads_end()),
      m_output_shape(pad->output(0).get_shape()) {
    if (pad->inputs().size() == 4) {
        auto const_node =
            std::dynamic_pointer_cast<op::Constant>(pad->input(3).get_source_output().get_node_shared_ptr());
        if (!const_node) {
            throw ngraph_error("Pad " + pad->get_friendly_name() + " with not constant pad_value is not allowed");
        }
        if (!util::get_single_value(const_node, m_pad_value)) {
            throw ngraph_error("Unsupported pad value");
        }
    }
    constructor_validate_and_infer_types();
}

void op::PadIE::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), m_output_shape);
}

shared_ptr<Node> op::PadIE::clone_with_new_inputs(const OutputVector& new_args) const {
    return nullptr;
}
