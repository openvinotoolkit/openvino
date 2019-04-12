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
#include "program_impl.h"

using namespace cldnn;

void mark_nodes::run(program_impl& p) {
    mark_constants(p);
    mark_data_flow(p);
}

void mark_nodes::mark_constants(program_impl& p)
{
    for (auto& node : p.get_processing_order())
    {
        p.mark_if_constant(*node);
    }
}

void mark_nodes::mark_data_flow(program_impl& p)
{
    for (auto const& node : p.get_processing_order())
    {
        p.mark_if_data_flow(*node);
    }
}
