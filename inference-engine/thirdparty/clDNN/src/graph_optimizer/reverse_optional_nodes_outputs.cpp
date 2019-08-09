/*
// Copyright (c) 2019 Intel Corporation
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
#include "program_helpers.h"
#include "lstm_dynamic_timeloop_inst.h"

#include <iterator>

using namespace cldnn;


/*
    Pass made for nodes, which has optional outputs (and had to reverse connections so
    the processing order was valid).
*/
void reverse_optional_nodes_outputs::run(program_impl& p) {
    auto node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto& node = (*node_itr++);
        if (node->is_type<lstm_dynamic_timeloop>()) {
            auto& typed_node = node->as<lstm_dynamic_timeloop>();
            typed_node.reverse_optional_outputs_connections();
        }
    }
}
