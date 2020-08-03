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

#include "ngraph/pass/validate.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/itt.hpp"

using namespace ngraph;

bool pass::Validate::run_on_function(std::shared_ptr<Function> f)
{
    f->validate_nodes_and_infer_types();
    return false;
}
