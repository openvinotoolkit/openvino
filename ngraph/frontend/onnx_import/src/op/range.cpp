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

#include <memory>

#include "default_opset.hpp"
#include "op/round.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector range(const Node& node)
                {
                    const Output<ngraph::Node> start{node.get_ng_inputs().at(0)};
                    const Output<ngraph::Node> stop{node.get_ng_inputs().at(1)};
                    const Output<ngraph::Node> step{node.get_ng_inputs().at(2)};
                    return {std::make_shared<default_opset::Range>(
                        start, stop, step, node.get_ng_inputs().at(0).get_element_type())};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
