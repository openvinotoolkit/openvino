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

#pragma once

#include "ngraph/log.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace pass
    {
        class ConstantFolding;
        bool revalidate_and_ensure_static(std::shared_ptr<ngraph::Node> n);
    }
}

class NGRAPH_API ngraph::pass::ConstantFolding : public ngraph::pass::GraphRewrite
{
public:
    ConstantFolding(const ngraph::BuildNodeExecutorMap& cfmap = ngraph::BuildNodeExecutorMap())
        : GraphRewrite()
    {
        m_cfmap = cfmap;
        m_enable_shape_inference = true;
        construct_constant_quantize();
        construct_constant_dequantize();
        construct_constant_convert();
        construct_constant_arithmetic_reduction();
        construct_constant_logical_reduction();
        construct_constant_gather_with_subgraph();
        construct_constant_scatter_elements_update();
        construct_constant_select();
        construct_constant_one_hot();
        construct_constant_default();
    }

private:
    void construct_constant_quantize();
    void construct_constant_dequantize();
    void construct_constant_convert();
    void construct_constant_arithmetic_reduction();
    void construct_constant_logical_reduction();
    void construct_constant_gather_with_subgraph();
    void construct_constant_scatter_elements_update();
    void construct_constant_select();
    void construct_constant_one_hot();
    void construct_constant_default();

    bool cf_is_disabled(const std::shared_ptr<Node>&);

    ngraph::BuildNodeExecutorMap m_cfmap;
};
