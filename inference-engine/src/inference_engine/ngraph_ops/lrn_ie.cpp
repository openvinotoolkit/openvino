//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <memory>
#include <string>

#include "lrn_ie.hpp"

#include "ngraph/op/lrn.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

op::LRN_IE::LRN_IE(const ngraph::Output<ngraph::Node> &arg,
                   double alpha,
                   double beta,
                   double bias,
                   size_t size,
                   std::string region)
        : Op({arg})
        , m_alpha(alpha)
        , m_beta(beta)
        , m_bias(bias)
        , m_size(size)
        , m_region(region) {
    constructor_validate_and_infer_types();
}

void op::LRN_IE::validate_and_infer_types() {
    element::Type arg_type = get_input_element_type(0);
    PartialShape arg_shape = get_input_partial_shape(0);
    set_output_type(0, arg_type, arg_shape);

    const PartialShape& input_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_shape.rank().is_dynamic() ||
                          static_cast<size_t>(input_shape.rank()) >= 3,
                          "Argument must have rank >= 3 (argument shape: ",
                          input_shape,
                          ").");
}

shared_ptr<Node> op::LRN_IE::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::LRN_IE>(new_args.at(0), m_alpha, m_beta, m_bias, m_size, m_region);
}
