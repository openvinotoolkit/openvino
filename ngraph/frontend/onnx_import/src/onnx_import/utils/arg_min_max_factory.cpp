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

#include "onnx_import/utils/arg_min_max_factory.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/validation_util.hpp"
#include "onnx_import/default_opset.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace utils
        {
            ArgMinMaxFactory::ArgMinMaxFactory(const Node& node)
                : m_keep_dims{node.get_attribute_value<std::int64_t>("keepdims", 1)}
                , m_axis{node.get_attribute_value<std::int64_t>("axis", 0)}
            {
                m_input_node = node.get_ng_inputs().at(0);
            }

            std::shared_ptr<ngraph::Node> ArgMinMaxFactory::make_arg_max() const
            {
                return make_topk_subgraph(default_opset::TopK::Mode::MAX);
            }

            std::shared_ptr<ngraph::Node> ArgMinMaxFactory::make_arg_min() const
            {
                return make_topk_subgraph(default_opset::TopK::Mode::MIN);
            }

            std::shared_ptr<ngraph::Node>
                ArgMinMaxFactory::make_topk_subgraph(default_opset::TopK::Mode mode) const
            {
                const auto k_node =
                    default_opset::Constant::create(ngraph::element::i64, Shape{}, {1});
                const auto topk = std::make_shared<default_opset::TopK>(
                    m_input_node, k_node, m_axis, mode, default_opset::TopK::SortType::NONE);

                if (m_keep_dims == 0)
                {
                    const auto axis_to_remove =
                        default_opset::Constant::create(element::u64, Shape{}, {topk->get_axis()});
                    const auto reshaped_indices =
                        std::make_shared<default_opset::Squeeze>(topk->output(1), axis_to_remove);

                    return std::make_shared<default_opset::Convert>(reshaped_indices, element::i64);
                }
                return std::make_shared<default_opset::Convert>(topk->output(1), element::i64);
            }
        }
    }
}
