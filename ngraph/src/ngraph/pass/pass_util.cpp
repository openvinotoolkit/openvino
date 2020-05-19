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

#include "ngraph/pass/pass_util.hpp"
#include "ngraph/log.hpp"

using namespace std;
using namespace ngraph;

std::function<bool(std::shared_ptr<Node>)> ngraph::pass::get_no_fan_out_function()
{
    auto ret_fun = [](std::shared_ptr<Node> n) {
        auto users = n->get_users(true);
        std::set<std::shared_ptr<Node>> user_set(users.begin(), users.end());
        size_t num_unique_users = user_set.size();
        if (num_unique_users == 1)
        {
            return true;
        }
        else
        {
            NGRAPH_DEBUG << n->get_name() << " has fan out\n";
            return false;
        }
    };

    return ret_fun;
}
