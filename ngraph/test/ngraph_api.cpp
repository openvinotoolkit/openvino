//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

using namespace ngraph;
using namespace std;

TEST(ngraph_api, parse_version)
{
    size_t major;
    size_t minor;
    size_t patch;
    string extra;

    {
        string version = "0.25.1-rc.0+7c32240";
        parse_version_string(version, major, minor, patch, extra);
        EXPECT_EQ(0, major);
        EXPECT_EQ(25, minor);
        EXPECT_EQ(1, patch);
        EXPECT_STREQ(extra.c_str(), "-rc.0+7c32240");
    }

    {
        string version = "v0.25.1-rc.0+7c32240";
        parse_version_string(version, major, minor, patch, extra);
        EXPECT_EQ(0, major);
        EXPECT_EQ(25, minor);
        EXPECT_EQ(1, patch);
        EXPECT_STREQ(extra.c_str(), "-rc.0+7c32240");
    }

    {
        string version = "0.25.1+7c32240";
        parse_version_string(version, major, minor, patch, extra);
        EXPECT_EQ(0, major);
        EXPECT_EQ(25, minor);
        EXPECT_EQ(1, patch);
        EXPECT_STREQ(extra.c_str(), "+7c32240");
    }

    {
        string version = "0.25.1";
        parse_version_string(version, major, minor, patch, extra);
        EXPECT_EQ(0, major);
        EXPECT_EQ(25, minor);
        EXPECT_EQ(1, patch);
        EXPECT_STREQ(extra.c_str(), "");
    }

    {
        string version = "x0.25.1";
        EXPECT_THROW(parse_version_string(version, major, minor, patch, extra), runtime_error);
    }

    {
        string version = "0x.25.1";
        EXPECT_THROW(parse_version_string(version, major, minor, patch, extra), runtime_error);
    }

    {
        string version = ".25.1";
        EXPECT_THROW(parse_version_string(version, major, minor, patch, extra), runtime_error);
    }

    {
        string version = "123.456";
        EXPECT_THROW(parse_version_string(version, major, minor, patch, extra), runtime_error);
    }

    {
        string version = "";
        EXPECT_THROW(parse_version_string(version, major, minor, patch, extra), runtime_error);
    }

    {
        string version = "this is a test";
        EXPECT_THROW(parse_version_string(version, major, minor, patch, extra), runtime_error);
    }
}

TEST(ngraph_api, version)
{
    string version_label = NGRAPH_VERSION_LABEL;
    size_t expected_major;
    size_t expected_minor;
    size_t expected_patch;
    string expected_extra;
    parse_version_string(
        version_label, expected_major, expected_minor, expected_patch, expected_extra);

    size_t actual_major;
    size_t actual_minor;
    size_t actual_patch;
    string actual_extra;
    get_version(actual_major, actual_minor, actual_patch, actual_extra);
    EXPECT_EQ(expected_major, actual_major);
    EXPECT_EQ(expected_minor, actual_minor);
    EXPECT_EQ(expected_patch, actual_patch);
    EXPECT_STREQ(expected_extra.c_str(), actual_extra.c_str());
}
