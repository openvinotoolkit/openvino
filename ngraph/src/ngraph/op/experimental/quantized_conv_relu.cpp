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

#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/get_output_element.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::QuantizedConvolutionRelu::type_info;

op::QuantizedConvolutionRelu::QuantizedConvolutionRelu(const Output<Node>& data_batch,
                                                       const Output<Node>& filters,
                                                       const Strides& window_movement_strides,
                                                       const Strides& window_dilation_strides,
                                                       const CoordinateDiff& padding_below,
                                                       const CoordinateDiff& padding_above,
                                                       const Strides& data_dilation_strides,
                                                       const Output<Node>& scale)
    : Op({data_batch, filters, scale})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
{
    constructor_validate_and_infer_types();

    auto& data_batch_shape = data_batch.get_shape();
    auto& filters_shape = filters.get_shape();

    set_output_type(0,
                    element::u8,
                    util::infer_convolution_output_shape(this,
                                                         data_batch_shape,
                                                         filters_shape,
                                                         window_movement_strides,
                                                         window_dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         data_dilation_strides,
                                                         0, /* batch_axis_data,              */
                                                         1, /* input_channel_axis_data,      */
                                                         1, /* input_channel_axis_filters,   */
                                                         0, /* output_channel_axis_filters,  */
                                                         0, /* batch_axis_result,            */
                                                         1  /* output_channel_axis_result,   */
                                                         ));
}

shared_ptr<Node>
    op::QuantizedConvolutionRelu::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return shared_ptr<Node>(new QuantizedConvolutionRelu(new_args.at(0),
                                                         new_args.at(1),
                                                         get_window_movement_strides(),
                                                         get_window_dilation_strides(),
                                                         get_padding_below(),
                                                         get_padding_above(),
                                                         get_data_dilation_strides(),
                                                         new_args.at(2)));
}
