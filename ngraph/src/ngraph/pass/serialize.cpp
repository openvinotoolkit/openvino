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

#include <fstream>

#include "ngraph/file_util.hpp"
#include "ngraph/pass/serialize.hpp"
#include "ngraph/util.hpp"
#ifndef NGRAPH_JSON_DISABLE
#include "ngraph/serializer.hpp"
#include "nlohmann/json.hpp"
#endif

using namespace std;
using namespace ngraph;

pass::Serialization::Serialization(const string& name)
    : m_name{name}
{
}

bool pass::Serialization::run_on_module(vector<shared_ptr<Function>>& functions)
{
#ifndef NGRAPH_JSON_DISABLE
    // serializing the outermost functions
    // also implicitly serializes any inner functions
    serialize(m_name, functions.at(0), 4);
#endif
    return false;
}
