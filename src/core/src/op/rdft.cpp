//*****************************************************************************
// Copyright 2017-2022 Intel Corporation
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

#include "ngraph/op/rdft.hpp"

#include <algorithm>
#include <memory>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/host_tensor.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v9::RDFT);

op::v9::RDFT::RDFT(const Output<Node>& data, const Output<Node>& axes) : FFTBase(data, axes) {
    constructor_validate_and_infer_types();
}

op::v9::RDFT::RDFT(const Output<Node>& data, const Output<Node>& axes, const Output<Node>& signal_size)
    : FFTBase(data, axes, signal_size) {
    constructor_validate_and_infer_types();
}

bool op::v9::RDFT::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v9_RDFT_visit_attributes);
    return true;
}

std::shared_ptr<Node> op::v9::RDFT::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v9_RDFT_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2 || new_args.size() == 3, "Number of inputs must be 2 or 3");

    if (new_args.size() == 2) {
        return std::make_shared<op::v9::RDFT>(new_args.at(0), new_args.at(1));
    }

    return std::make_shared<op::v9::RDFT>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void op::v9::RDFT::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v9_RDFT_validate_and_infer_types);

    size_t num_of_inputs = get_input_size();
    NODE_VALIDATION_CHECK(this, num_of_inputs == 2 || num_of_inputs == 3, "FFT op must have 2 or 3 inputs.");

    element::Type input_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          input_et == element::f32 || input_et == element::f16 || input_et == element::bf16,
                          "RFFT op input element type must be f32, f16, or bf16");

    element::Type axes_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          axes_et == element::i64 || axes_et == element::i32,
                          "RFFT op axes element type must be i32 or i64");

    if (num_of_inputs == 3) {
        element::Type signal_size_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              signal_size_et == element::i64 || signal_size_et == element::i32,
                              "RFFT op signal_size element type must be i32 or i64");
    }

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
    std::vector<ov::PartialShape> input_shapes;

    const auto& data = get_input_partial_shape(0);
    const auto& axes = get_input_partial_shape(1);
    if (input_values().size() == 2) {
        input_shapes = {data, axes};
    } else {
        const auto& signal_size = get_input_partial_shape(2);
        input_shapes = {data, axes, signal_size};
    }

    rdft_shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}
