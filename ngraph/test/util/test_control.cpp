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
#include <unordered_map>
#include <unordered_set>

#include "ngraph/log.hpp"
#include "ngraph/util.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static unordered_set<string>& get_blacklist(const string& backend)
{
    static unordered_map<string, unordered_set<string>> s_blacklists;
    return s_blacklists[backend];
}

string ngraph::prepend_disabled(const string& backend_name,
                                const string& test_name,
                                const string& manifest)
{
    string rc = test_name;
    unordered_set<string>& blacklist = get_blacklist(backend_name);
    if (blacklist.empty() && !manifest.empty())
    {
        ifstream f(manifest);
        string line;
        while (getline(f, line))
        {
            size_t pound_pos = line.find('#');
            line = (pound_pos > line.size()) ? line : line.substr(0, pound_pos);
            line = trim(line);
            if (line.size() > 1)
            {
                blacklist.insert(line);
            }
        }
    }
    string compound_test_name = backend_name + "." + test_name;
    if (blacklist.find(test_name) != blacklist.end() ||
        blacklist.find(compound_test_name) != blacklist.end())
    {
        rc = "DISABLED_" + test_name;
    }
    return rc;
}

string ngraph::combine_test_backend_and_case(const string& backend_name,
                                             const string& test_casename)
{
    if (backend_name == test_casename)
    {
        return backend_name;
    }
    else
    {
        return backend_name + "/" + test_casename;
    }
}
