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

#include <memory>

#include "ngraph/builder/quantized_conv_builder.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace builder
    {
        shared_ptr<Node> QuantizedConvolutionBuilder(const Output<Node>& input,
                                                     const Output<Node>& filters,
                                                     const Strides& window_movement_strides,
                                                     const Strides& window_dilation_strides,
                                                     const CoordinateDiff& padding_below,
                                                     const CoordinateDiff& padding_above,
                                                     const Strides& data_dilation_strides,
                                                     const Output<Node>& min_input,
                                                     const Output<Node>& max_input,
                                                     const Output<Node>& min_filter,
                                                     const Output<Node>& max_filter,
                                                     const Output<Node>& min_output,
                                                     const Output<Node>& max_output,
                                                     const ngraph::element::Type& output_type,
                                                     const ngraph::AxisSet& input_axes,
                                                     const ngraph::AxisSet& filter_axes,
                                                     const ngraph::AxisSet& output_axes)
        {
            auto input_scale =
                quantization_utils::get_scale(min_input, max_input, input.get_element_type());
            auto filter_scale =
                quantization_utils::get_scale(min_filter, max_filter, filters.get_element_type());
            auto output_scale = quantization_utils::get_scale(min_output, max_output, output_type);

            // TODO: Check for this later
            // For Builders the zero point is assumed to be zero (for now)
            auto input_zero_point = op::Constant::create(input.get_element_type(), Shape{}, {0});
            auto filter_zero_point = op::Constant::create(filters.get_element_type(), Shape{}, {0});

            return make_shared<op::QuantizedConvolution>(
                       input,
                       filters,
                       window_movement_strides,
                       window_dilation_strides,
                       padding_below,
                       padding_above,
                       data_dilation_strides,
                       input_scale,
                       input_zero_point,
                       filter_scale,
                       filter_zero_point,
                       output_scale,
                       filter_zero_point, // output type will be same as filter
                       output_type,
                       input_axes,
                       filter_axes,
                       output_axes)
                ->add_provenance_group_members_above({input,
                                                      filters,
                                                      min_input,
                                                      max_input,
                                                      min_filter,
                                                      max_filter,
                                                      min_output,
                                                      max_output});
        }
    }
}
