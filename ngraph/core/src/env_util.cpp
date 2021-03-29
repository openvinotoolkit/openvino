// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>

#include "ngraph/env_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;

std::string ngraph::getenv_string(const char* env_var)
{
    const char* env_p = ::getenv(env_var);
    return env_p != nullptr ? string(env_p) : "";
}

int32_t ngraph::getenv_int(const char* env_var, int32_t default_value)
{
    const char* env_p = ::getenv(env_var);
    int32_t env = default_value;
    // If env_var is not "" or undefined
    if (env_p && *env_p)
    {
        errno = 0;
        char* err;
        env = strtol(env_p, &err, 0);
        // if conversion leads to an overflow
        if (errno)
        {
            std::stringstream ss;
            ss << "Environment variable \"" << env_var << "\"=\"" << env_p
               << "\" converted to different value \"" << env << "\" due to overflow." << std::endl;
            throw runtime_error(ss.str());
        }
        // if syntax error is there - conversion will still happen
        // but warn user of syntax error
        if (*err)
        {
            std::stringstream ss;
            ss << "Environment variable \"" << env_var << "\"=\"" << env_p
               << "\" converted to different value \"" << env << "\" due to syntax error \"" << err
               << '\"' << std::endl;
            throw runtime_error(ss.str());
        }
    }
    else
    {
        NGRAPH_DEBUG << "Environment variable (" << env_var << ") empty or undefined, "
                     << " defaulted to " << default_value << " here.";
    }
    return env;
}

bool ngraph::getenv_bool(const char* env_var, bool default_value)
{
    string value = to_lower(getenv_string(env_var));
    set<string> off = {"0", "false", "off"};
    set<string> on = {"1", "true", "on"};
    bool rc;
    if (value == "")
    {
        rc = default_value;
    }
    else if (off.find(value) != off.end())
    {
        rc = false;
    }
    else if (on.find(value) != on.end())
    {
        rc = true;
    }
    else
    {
        stringstream ss;
        ss << "environment variable '" << env_var << "' value '" << value
           << "' invalid. Must be boolean.";
        throw runtime_error(ss.str());
    }
    return rc;
}
