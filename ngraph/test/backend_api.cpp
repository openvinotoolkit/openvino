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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/util.hpp"
#include "runtime/backend.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(backend_api, registered_devices)
{
    vector<string> devices = runtime::Backend::get_registered_devices();
    EXPECT_GE(devices.size(), 0);

    EXPECT_TRUE(find(devices.begin(), devices.end(), "INTERPRETER") != devices.end());
}

TEST(backend_api, invalid_name)
{
    ASSERT_ANY_THROW(ngraph::runtime::Backend::create("COMPLETELY-BOGUS-NAME"));
}

TEST(backend_api, config)
{
    auto backend = runtime::Backend::create("INTERPRETER");
    string error;
    string message = "hello";
    map<string, string> config = {{"test_echo", message}};
    EXPECT_TRUE(backend->set_config(config, error));
    EXPECT_STREQ(error.c_str(), message.c_str());
    EXPECT_FALSE(backend->set_config({}, error));
    EXPECT_STREQ(error.c_str(), "");
}

TEST(backend_api, DISABLED_config_unsupported)
{
    auto backend = runtime::Backend::create("NOP");
    string error;
    string message = "hello";
    map<string, string> config = {{"test_echo", message}};
    EXPECT_FALSE(backend->set_config(config, error));
    EXPECT_FALSE(error == "");
}
