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

#include "default_opset.hpp"
#include "ngraph/op/normalize_l2.hpp"
#include "op/org.openvinotoolkit/normalize.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector swish(const Node& node)
                {
                    OutputVector ng_inputs{node.get_ng_inputs()};

                    Output<ngraph::Node> beta;
                    if (ng_inputs.size() > 1)
                    {
                        beta = ngraph::onnx_import::reshape::interpret_as_scalar(ng_inputs.at(1));
                    }
                    else
                    {
                        beta = default_opset::Constant::create(element::f32, Shape{}, {1.0});
                    }

                    return {std::make_shared<default_opset::Swish>(ng_inputs.at(0), beta)};
                }

            } // namespace set_1
        }     // namespace op

    } // namespace onnx_import

} // namespace ngraph
