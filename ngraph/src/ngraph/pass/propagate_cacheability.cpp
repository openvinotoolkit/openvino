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

#include "ngraph/pass/propagate_cacheability.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/util/op_annotations.hpp"

using namespace std;
using namespace ngraph;

bool pass::PropagateCacheability::run_on_function(shared_ptr<Function> function)
{
    for (auto& node : function->get_ordered_ops())
    {
        if (node->is_op())
        {
            auto op = static_pointer_cast<op::Op>(node);
            NGRAPH_DEBUG << "propagate cacheability: node is " << node->get_name();
            auto op_annotations = op->get_op_annotations();
            if (!op_annotations)
            {
                NGRAPH_DEBUG << "propagate cacheability: create op_annotations";
                op_annotations = op_annotations_factory();
                op->set_op_annotations(op_annotations);
            }
            if (node->is_parameter())
            {
                auto parameter = static_pointer_cast<op::Parameter>(node);
                op_annotations->set_cacheable(parameter->get_cacheable());
                NGRAPH_DEBUG << "propagate cacheability: cacheability is "
                             << parameter->get_cacheable();
            }
            else
            {
                bool cacheable = true;
                for (auto input : node->inputs())
                {
                    auto input_value_node = input.get_source_output().get_node_shared_ptr();
                    NGRAPH_DEBUG << "propagate cacheability: arg is " << *input_value_node;
                    if (input_value_node->is_op())
                    {
                        auto arg_op = static_pointer_cast<op::Op>(input_value_node);
                        auto arg_op_annotations = arg_op->get_op_annotations();
                        NGRAPH_CHECK(arg_op_annotations);
                        if (!arg_op_annotations->is_cacheable())
                        {
                            cacheable = false;
                            break;
                        }
                    }
                }
                NGRAPH_DEBUG << "propagate cacheability: cacheability is " << cacheable;
                op_annotations->set_cacheable(cacheable);
            }
        }
    }
    return false;
}
