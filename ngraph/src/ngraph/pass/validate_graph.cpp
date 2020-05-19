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

#include "ngraph/pass/validate_graph.hpp"

using namespace std;
using namespace ngraph;

bool pass::ValidateGraph::run_on_module(vector<shared_ptr<Function>>& functions)
{
    for (shared_ptr<Function> f : functions)
    {
        validate_parameters(*f);
    }
    return false;
}

void pass::ValidateGraph::validate_parameters(const Function& function)
{
    auto parameters = function.get_parameters();
    for (auto node : function.get_ops())
    {
        shared_ptr<op::Parameter> p = as_type_ptr<op::Parameter>(node);
        if (nullptr != p)
        {
            auto it = find_if(parameters.begin(),
                              parameters.end(),
                              [p](shared_ptr<op::Parameter> q) { return (p == q); });
            if (it == parameters.end())
            {
                throw ngraph_error("Function references undeclared parameter");
            }
        }
    }
}
