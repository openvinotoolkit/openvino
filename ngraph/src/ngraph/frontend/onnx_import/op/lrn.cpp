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
#include "lrn.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector lrn(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    double alpha = node.get_attribute_value<double>("alpha", 1e-4);
                    double beta = node.get_attribute_value<double>("beta", 0.75);
                    double bias = node.get_attribute_value<double>("bias", 1);
                    size_t size = node.get_attribute_value<size_t>("size");

                    return {std::make_shared<default_opset::LRN>(data, alpha, beta, bias, size)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
