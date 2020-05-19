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

#include <cstdio>
#include <string>

#include "ngraph/distributed/null.hpp"
#include "ngraph/except.hpp"

const std::string& ngraph::distributed::Null::get_name() const
{
    return m_name;
}

int ngraph::distributed::Null::get_size()
{
    return 0;
}

int ngraph::distributed::Null::get_rank()
{
    return 0;
}

void ngraph::distributed::Null::all_reduce(void*, void*, element::Type_t, reduction::Type, size_t)
{
    throw ngraph_error("Distributed Library not supported/mentioned");
}

void ngraph::distributed::Null::broadcast(void*, element::Type_t, size_t, int)
{
    throw ngraph_error("Distributed Library not supported/mentioned");
}

void ngraph::distributed::Null::recv(void*, element::Type_t, size_t, int)
{
    throw ngraph_error("Distributed Library not supported/mentioned");
}

void ngraph::distributed::Null::send(const void*, element::Type_t, size_t, int)
{
    throw ngraph_error("Distributed Library not supported/mentioned");
}
