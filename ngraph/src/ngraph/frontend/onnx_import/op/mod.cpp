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
#include "exceptions.hpp"
#include "mod.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/fused/mod.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector mod(const Node& node)
                {
                    std::shared_ptr<ngraph::Node> dividend{node.get_ng_inputs().at(0)};
                    std::shared_ptr<ngraph::Node> divisor{node.get_ng_inputs().at(1)};

                    std::int64_t fmod = node.get_attribute_value<std::int64_t>("fmod", 0);
                    CHECK_VALID_NODE(
                        node, fmod == 1, "Only 'fmod=1' mode is supported for mod operator.");

                    return {std::make_shared<default_opset::Mod>(dividend, divisor)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
