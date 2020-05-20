//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <functional>

#include "ngraph/shape.hpp"
#include "quantized_dot_bias.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::QuantizedDotBias::type_info;

op::QuantizedDotBias::QuantizedDotBias(const Output<Node>& data,
                                       const Output<Node>& weights,
                                       const Output<Node>& bias,
                                       const Output<Node>& scale,
                                       bool requantize,
                                       bool with_relu)
    : Op({data, weights, bias, scale})
    , m_requantize(requantize)
    , m_with_relu(with_relu)
{
    constructor_validate_and_infer_types();

    auto& data_shape = data.get_shape();
    auto& weights_shape = weights.get_shape();
    auto& bias_shape = bias.get_shape();
    NODE_VALIDATION_CHECK(this,
                          data_shape.size() == 2 && weights_shape.size() == 2 &&
                              data_shape[1] == weights_shape[1],
                          "only valid tensors of rank 2 supported. data ",
                          data_shape,
                          " weights ",
                          weights_shape);
    NODE_VALIDATION_CHECK(this,
                          bias_shape.size() == 1 && bias_shape[0] == weights_shape[0],
                          "invalid bias ",
                          bias_shape);

    auto output_et = requantize ? (with_relu ? element::u8 : element::i8) : element::f32;
    set_output_type(0, output_et, Shape{data_shape[0], weights_shape[0]});
}
