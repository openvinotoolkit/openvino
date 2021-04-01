//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "op/org.openvinotoolkit/deformable_conv_2d.hpp"
#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/deformable_convolution.hpp"
#include "utils/convpool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        OutputVector op::set_1::deformable_conv_2d(const Node& node)
        {
            const OutputVector& inputs = node.get_ng_inputs();
            const auto strides = convpool::get_strides(node);
            const auto dilations = convpool::get_dilations(node);
            const auto paddings = convpool::get_pads(node);

            const auto group = node.get_attribute_value<int64_t>("group", 1);
            const auto deformable_groups =
                node.get_attribute_value<int64_t>("deformable_groups", 1);
            const auto auto_pad_type = convpool::get_auto_pad(node);

            return {std::make_shared<default_opset::DeformableConvolution>(inputs.at(0),
                                                                           inputs.at(1),
                                                                           inputs.at(2),
                                                                           strides,
                                                                           paddings.first,
                                                                           paddings.second,
                                                                           dilations,
                                                                           auto_pad_type,
                                                                           group,
                                                                           deformable_groups)};
        }
    } // namespace onnx_import
} // namespace ngraph
