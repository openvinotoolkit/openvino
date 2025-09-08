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

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/deformable_convolution.hpp"
#include "utils/convpool.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace org_openvinotoolkit {
namespace opset_1 {
ov::OutputVector deformable_conv_2d(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector& inputs = node.get_ov_inputs();
    const auto strides = convpool::get_strides(node);
    const auto dilations = convpool::get_dilations(node);
    const auto paddings = convpool::get_pads(node);

    const auto group = node.get_attribute_value<int64_t>("group", 1);
    const auto deformable_groups = node.get_attribute_value<int64_t>("deformable_groups", 1);
    const auto auto_pad_type = convpool::get_auto_pad(node);

    if (inputs.size() == 3) {
        return {std::make_shared<v8::DeformableConvolution>(inputs.at(0),
                                                            inputs.at(1),
                                                            inputs.at(2),
                                                            strides,
                                                            paddings.first,
                                                            paddings.second,
                                                            dilations,
                                                            auto_pad_type,
                                                            group,
                                                            deformable_groups)};
    } else if (inputs.size() == 4) {
        return {std::make_shared<v8::DeformableConvolution>(inputs.at(0),
                                                            inputs.at(1),
                                                            inputs.at(2),
                                                            inputs.at(3),
                                                            strides,
                                                            paddings.first,
                                                            paddings.second,
                                                            dilations,
                                                            auto_pad_type,
                                                            group,
                                                            deformable_groups)};
    } else {
        FRONT_END_GENERAL_CHECK(false, "Invalid number of inputs");
    }
}
ONNX_OP("DeformableConv2D", OPSET_SINCE(1), org_openvinotoolkit::opset_1::deformable_conv_2d, OPENVINO_ONNX_DOMAIN);
}  // namespace opset_1
}  // namespace org_openvinotoolkit
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
