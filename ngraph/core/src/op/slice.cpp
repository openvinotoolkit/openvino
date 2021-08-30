// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/slice.hpp"

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v8::Slice, "Slice", 8);

op::v8::Slice::Slice(const Output<Node>& data,
                     const Output<Node>& start,
                     const Output<Node>& stop,
                     const Output<Node>& step)
    : Op({data, start, stop, step}) {
    constructor_validate_and_infer_types();
}

op::v8::Slice::Slice(const Output<Node>& data,
                     const Output<Node>& start,
                     const Output<Node>& stop,
                     const Output<Node>& step,
                     const Output<Node>& axes)
    : Op({data, start, stop, step, axes}) {
    constructor_validate_and_infer_types();
}

bool op::v8::Slice::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v8_Slice_visit_attributes);
    return true;
}

void op::v8::Slice::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v8_Slice_validate_and_infer_types);

    const PartialShape& data_shape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), data_shape);
}

shared_ptr<Node> op::v8::Slice::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v8_Slice_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 4) {
        return std::make_shared<v8::Slice>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
    } else {
        return std::make_shared<v8::Slice>(new_args.at(0),
                                           new_args.at(1),
                                           new_args.at(2),
                                           new_args.at(3),
                                           new_args.at(4));
    }
}
