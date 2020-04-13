/*
// Copyright (c) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_node.h"
#include "layout_optimizer.h"
#include "program_impl.h"
#include "program_helpers.h"
#include <vector>
#include <memory>
#include <list>
#include <map>
#include <set>

using namespace cldnn;

void basic_memory_dependencies::run(program_impl& p) {
    auto itr = p.get_processing_order().begin();
    std::vector<primitive_id> past_outputs;
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr;
        itr++;

        // data primitive can't be reused
        if (node->is_type<data>())
            continue;

        // add my dependencies to restriction list (can't share input.output buffers)
        for (auto it : node->get_dependencies()) {
            add_memory_dependency(node, it);
            add_memory_dependency(it, node);
        }

        // Note we iterate over processing order, it means if primitve has processing num greater than any of outputs,
        // this output has to land on the primitve restriction list. Otherwise memory reuse can corrupt final results.
        node->add_memory_dependency(past_outputs);
        // if current node is an output add it to the outputs list after restriction.
        if (node->is_output())
            past_outputs.push_back(node->id());
    }
}
