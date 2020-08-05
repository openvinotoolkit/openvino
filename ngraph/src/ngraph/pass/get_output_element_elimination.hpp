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

#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class GetOutputElementElimination;
    }
}

NGRAPH_SUPPRESS_DEPRECATED_START
class NGRAPH_API ngraph::pass::GetOutputElementElimination : public NodePass
{
public:
    bool run_on_node(std::shared_ptr<Node> node) override;
};
NGRAPH_SUPPRESS_DEPRECATED_END
