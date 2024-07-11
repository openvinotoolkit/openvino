// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/env_util.hpp"

#include <set>
#include <sstream>

#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"

std::string ov::util::getenv_string(const char* env_var) {
    const char* env_p = ::getenv(env_var);
    return env_p != nullptr ? std::string(env_p) : "";
}

int32_t ov::util::getenv_int(const char* env_var, int32_t default_value) {
    const char* env_p = ::getenv(env_var);
    int32_t env = default_value;
    // If env_var is not "" or undefined
    if (env_p && *env_p) {
        errno = 0;
        char* err;
        env = strtol(env_p, &err, 0);
        // if conversion leads to an overflow
        if (errno) {
            std::stringstream ss;
            ss << "Environment variable \"" << env_var << "\"=\"" << env_p << "\" converted to different value \""
               << env << "\" due to overflow." << std::endl;
            throw std::runtime_error(ss.str());
        }
        // if syntax error is there - conversion will still happen
        // but warn user of syntax error
        if (*err) {
            std::stringstream ss;
            ss << "Environment variable \"" << env_var << "\"=\"" << env_p << "\" converted to different value \""
               << env << "\" due to syntax error \"" << err << '\"' << std::endl;
            throw std::runtime_error(ss.str());
        }
    }
#ifdef ENABLE_OPENVINO_DEBUG
    else {
        OPENVINO_DEBUG("Environment variable (",
                       env_var,
                       ") empty or undefined, defaulted to ",
                       default_value,
                       " here.");
    }
#endif
    return env;
}

bool ov::util::getenv_bool(const char* env_var, bool default_value) {
    std::string value = ov::util::to_lower(ov::util::getenv_string(env_var));
    std::set<std::string> off = {"0", "false", "off"};
    std::set<std::string> on = {"1", "true", "on"};
    bool rc;
    if (value == "") {
        rc = default_value;
    } else if (off.find(value) != off.end()) {
        rc = false;
    } else if (on.find(value) != on.end()) {
        rc = true;
    } else {
        std::stringstream ss;
        ss << "environment variable '" << env_var << "' value '" << value << "' invalid. Must be boolean.";
        throw std::runtime_error(ss.str());
    }
    return rc;
}
