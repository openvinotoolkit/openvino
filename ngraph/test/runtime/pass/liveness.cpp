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

#include <exception>
#include <sstream>
#include <unordered_set>

#include "liveness.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

bool pass::Liveness::run_on_function(shared_ptr<Function> function)
{
    auto ops = function->get_ordered_ops();

    unordered_set<descriptor::Tensor*> persistent_tensors;
    unordered_set<descriptor::Tensor*> output_tensors;
    for (const shared_ptr<op::Parameter>& node : function->get_parameters())
    {
        for (auto& output : node->outputs())
        {
            descriptor::Tensor& tensor = output.get_tensor();
            persistent_tensors.insert(&tensor);
        }
    }
    for (const shared_ptr<op::Result>& node : function->get_results())
    {
        for (auto& output : node->outputs())
        {
            descriptor::Tensor& tensor = output.get_tensor();
            persistent_tensors.insert(&tensor);
            output_tensors.insert(&tensor);
        }
    }
    for (const shared_ptr<Node>& node : ops)
    {
        if (auto constant_node = as_type_ptr<op::Constant>(node))
        {
            for (auto& output : constant_node->outputs())
            {
                descriptor::Tensor& tensor = output.get_tensor();
                persistent_tensors.insert(&tensor);
            }
        }
    }

    unordered_set<descriptor::Tensor*> currently_live;
    for (auto it = ops.rbegin(); it != ops.rend(); it++)
    {
        const shared_ptr<Node>& node = *it;
        node->liveness_new_list.clear();
        node->liveness_free_list.clear();
        unordered_set<descriptor::Tensor*> input_tensor_decls;
        for (auto& input : node->inputs())
        {
            descriptor::Tensor& tensor = input.get_tensor();
            if (persistent_tensors.find(&tensor) == persistent_tensors.end())
            {
                input_tensor_decls.insert(&tensor);
            }
        }

        unordered_set<descriptor::Tensor*> output_tensor_decls;
        for (auto& output : node->outputs())
        {
            descriptor::Tensor& tensor = output.get_tensor();
            if (persistent_tensors.find(&tensor) == persistent_tensors.end())
            {
                output_tensor_decls.insert(&tensor);
            }
        }

        unordered_set<descriptor::Tensor*> free_tensor_decls;
        unordered_set<descriptor::Tensor*> new_tensor_decls;
        unordered_set<descriptor::Tensor*> all_tensor_decls = input_tensor_decls;
        all_tensor_decls.insert(output_tensor_decls.begin(), output_tensor_decls.end());

        for (descriptor::Tensor* tensor_decl : all_tensor_decls)
        {
            if (currently_live.find(tensor_decl) == currently_live.end())
            {
                // this is the last node that value is seen in
                // delete it at the end of the op
                currently_live.insert(tensor_decl);
                if (output_tensors.find(tensor_decl) == output_tensors.end())
                {
                    // Don't free output tensors
                    free_tensor_decls.insert(tensor_decl);
                }
            }
        }

        for (descriptor::Tensor* output_decl : output_tensor_decls)
        {
            auto currently_live_it = currently_live.find(output_decl);
            if (currently_live_it != currently_live.end())
            {
                new_tensor_decls.insert(output_decl);
                currently_live.erase(currently_live_it);
            }
        }
        node->liveness_free_list = free_tensor_decls;
        node->liveness_new_list = new_tensor_decls;
    }

    return false;
}
