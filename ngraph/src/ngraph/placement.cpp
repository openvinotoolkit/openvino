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

#include <deque>
#include <sstream>

#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/placement.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

std::string ngraph::placement_to_string(Placement placement)
{
    switch (placement)
    {
    case Placement::DEFAULT: return "DEFAULT";
    case Placement::INTERPRETER: return "INTERPRETER";
    case Placement::CPU: return "CPU";
    case Placement::GPU: return "GPU";
    case Placement::NNP: return "NNP";
    }
    throw runtime_error("unhandled placement type");
}
