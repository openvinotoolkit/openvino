// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn/runtime/debug_configuration.hpp"
#include <iostream>
#include <memory>

namespace cldnn {

const char *debug_configuration::prefix = "GPU_Debug: ";

// Default policy is that dump_configuration will override other configuration from IE.

#ifdef GPU_DEBUG_CONFIG
static void print_option(std::string option_name, std::string option_value) {
    GPU_DEBUG_COUT << "Config " << option_name << " = " << option_value << std::endl;
}

static void get_int_env(const std::string &var, int &val) {
    if (const auto env_var = std::getenv(var.c_str())) {
        val = std::stoi(env_var);
        print_option(var, std::to_string(val));
    }
}

static void get_str_env(const std::string &var, std::string &val) {
    if (const auto env_var = std::getenv(var.c_str())) {
        val = env_var;
        print_option(var, val);
    }
}

#endif

debug_configuration::debug_configuration()
        : verbose(0)
        , dump_graphs(std::string()) {
#ifdef GPU_DEBUG_CONFIG
    get_int_env("OV_GPU_Verbose", verbose);
    get_str_env("OV_GPU_DumpGraphs", dump_graphs);
#endif
}

const debug_configuration *debug_configuration::get_instance() {
    static std::unique_ptr<debug_configuration> instance(nullptr);
#ifdef GPU_DEBUG_CONFIG
    static std::mutex _m;
    std::lock_guard<std::mutex> lock(_m);
    if (nullptr == instance)
        instance.reset(new debug_configuration());
    return instance.get();
#else
    return nullptr;
#endif
}
} // namespace cldnn
