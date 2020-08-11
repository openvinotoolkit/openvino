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

#include "depth_to_space.hpp"
#include "onnx_import/default_opset.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector depth_to_space(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    NGRAPH_CHECK(data.get_shape().size() == 4, "Input must be 4-dimensional");

                    const auto mode = node.get_attribute_value<std::string>("mode", "DCR");
                    default_opset::DepthToSpace::DepthToSpaceMode ngraph_mode;
                    if (mode == "DCR")
                        ngraph_mode = default_opset::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
                    else if (mode == "CRD")
                        ngraph_mode = default_opset::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST;
                    else
                        NGRAPH_CHECK(false, "only 'DCR' and 'CRD' modes are supported");

                    const auto block_size = node.get_attribute_value<std::int64_t>("blocksize");
                    return OutputVector{std::make_shared<default_opset::DepthToSpace>(
                        data, ngraph_mode, block_size)};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
