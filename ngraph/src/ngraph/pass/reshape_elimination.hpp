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
#include "ngraph/pass/pass_util.hpp"

namespace ngraph
{
    namespace pass
    {
        class ReshapeElimination;
        class RecurrentReshapeElimination;
    }
}

class NGRAPH_API ngraph::pass::ReshapeElimination : public ngraph::pass::GraphRewrite
{
public:
    ReshapeElimination()
        : GraphRewrite()
    {
        construct_dot_transpose_pattern();
        construct_identity_reshape_pattern();
        construct_reshapex2_pattern();
    }

private:
    void construct_dot_transpose_pattern();
    void construct_identity_reshape_pattern();
    void construct_reshapex2_pattern();
};

class NGRAPH_API ngraph::pass::RecurrentReshapeElimination
    : public ngraph::pass::RecurrentGraphRewrite
{
public:
    RecurrentReshapeElimination()
        : RecurrentGraphRewrite()
    {
        construct_recurrent_reshape();
    }

private:
    void construct_recurrent_reshape();
};
