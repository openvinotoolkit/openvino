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
    ConstantFolding(const ngraph::BuildNodeExecutorMap& cfmap = ngraph::BuildNodeExecutorMap());

private:
    void copy_runtime_info_to_target_inputs(const std::shared_ptr<Node>& node,
                                            const Output<Node>& replacement);

    ngraph::BuildNodeExecutorMap m_cfmap;
};
