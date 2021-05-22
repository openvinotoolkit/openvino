// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/qlinear_matmul.hpp"
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
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
