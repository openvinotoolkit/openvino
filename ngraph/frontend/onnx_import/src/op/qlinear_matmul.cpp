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

#include "onnx_import/op/qlinear_matmul.hpp"
#include "ngraph/builder/matmul_factory.hpp"
#include "ngraph/log.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector qlinear_matmul(const Node& node)
                {
                    NGRAPH_SUPPRESS_DEPRECATED_START
                    auto ng_inputs = node.get_ng_inputs();
                    auto factory = builder::QLinearMatmulFactory(
                        (OutputVector(std::begin(ng_inputs), std::end(ng_inputs))));
                    std::size_t left_rank{ng_inputs.at(0).get_shape().size()};
                    std::size_t right_rank{ng_inputs.at(1).get_shape().size()};

                    if (left_rank == 0 || right_rank == 0)
                    {
                        NGRAPH_WARN
                            << (node) << " "
                            << "ONNX standard doesn't allow scalar operands, however nGraph "
                               "accepts them. Consider use of element-wise multiplication instead "
                               "to conform with ONNX standard.";
                    }
                    return factory.make_matmul_op();
                    NGRAPH_SUPPRESS_DEPRECATED_END
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
