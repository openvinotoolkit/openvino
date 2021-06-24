// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include "default_opset.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector gather_nd(const Node& node)
                {
                    const OutputVector ng_inputs{node.get_ng_inputs()};
                    const auto data = ng_inputs.at(0);
                    const auto indices = ng_inputs.at(1);
                    const auto batch_dims = node.get_attribute_value<int64_t>("batch_dims", 0);

                    return {std::make_shared<default_opset::GatherND>(data, indices, batch_dims)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
