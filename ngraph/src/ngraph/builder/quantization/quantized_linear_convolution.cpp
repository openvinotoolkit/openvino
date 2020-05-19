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

#include "ngraph/builder/quantization/quantized_linear_convolution.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace builder
    {
        namespace quantization
        {
            shared_ptr<Node> QuantizedLinearConvolutionBias(const Output<Node>& input,
                                                            const Output<Node>& filter,
                                                            const Output<Node>& bias,
                                                            const Strides& window_movement_strides,
                                                            const Strides& window_dilation_strides,
                                                            const CoordinateDiff& padding_below,
                                                            const CoordinateDiff& padding_above,
                                                            const Strides& data_dilation_strides,
                                                            const Output<Node>& input_scale,
                                                            const Output<Node>& filter_scale,
                                                            const Output<Node>& output_scale)
            {
                // TODO: need to establish cross-nGraph view of scale (mult or div)
                auto requantization_scale = (input_scale * filter_scale) / output_scale;

                auto mybias = bias;
                if (bias.get_element_type() != element::i32)
                {
                    const auto zero = make_constant(element::i32, input_scale.get_shape(), 0);
                    const AxisSet quantization_axes;
                    const auto bias_scale = input_scale * filter_scale;
                    op::Quantize::RoundMode round_mode =
                        op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

                    mybias = make_shared<op::Quantize>(
                        bias, bias_scale, zero, element::i32, quantization_axes, round_mode);
                }
                return make_shared<op::QuantizedConvolutionBias>(input,
                                                                 filter,
                                                                 mybias,
                                                                 window_movement_strides,
                                                                 window_dilation_strides,
                                                                 padding_below,
                                                                 padding_above,
                                                                 data_dilation_strides,
                                                                 requantization_scale,
                                                                 false)
                    ->add_provenance_group_members_above(
                        {input, filter, bias, input_scale, filter_scale, output_scale});
            }
        }
    }
}
