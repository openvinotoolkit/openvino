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

#include "default_opset.hpp"
#include "round.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector range(const Node& node)
                {
                    const std::shared_ptr<ngraph::Node> start{node.get_ng_inputs().at(0)};
                    const std::shared_ptr<ngraph::Node> stop{node.get_ng_inputs().at(1)};
                    const std::shared_ptr<ngraph::Node> step{node.get_ng_inputs().at(2)};
                    return {std::make_shared<default_opset::Range>(start, stop, step)};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
