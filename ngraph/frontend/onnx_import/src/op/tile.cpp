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
#include "onnx_import/core/node.hpp"
#include "op/tile.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector tile(const Node& node)
                {
                    auto input = node.get_ng_inputs().at(0);
                    auto repeats = node.get_ng_inputs().at(1);

                    // Workaround for backends which require repeats to be i64.
                    // Remove the following line when no longer needed.
                    repeats = std::make_shared<default_opset::Convert>(repeats, element::i64);

                    return {std::make_shared<default_opset::Tile>(input, repeats)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
