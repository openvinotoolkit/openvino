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
        , print_multi_kernel_perf(0)
        , disable_usm(0)
        , dump_graphs(std::string())
        , dump_layers_path(std::string())
        , dump_layers(std::string())
        , dump_layers_dst_only(0) {
#ifdef GPU_DEBUG_CONFIG
    get_int_env("OV_GPU_Verbose", verbose);
    get_int_env("OV_GPU_PrintMultiKernelPerf", print_multi_kernel_perf);
    get_int_env("OV_GPU_DisableUsm", disable_usm);
    get_str_env("OV_GPU_DumpGraphs", dump_graphs);
    get_str_env("OV_GPU_DumpLayersPath", dump_layers_path);
    get_str_env("OV_GPU_DumpLayers", dump_layers);
    get_int_env("OV_GPU_DumpLayersDstOnly", dump_layers_dst_only);
    if (dump_layers_path.length() > 0 && !disable_usm) {
        disable_usm = 1;
        GPU_DEBUG_COUT << "DisableUsm=1 because of DumpLayersPath" << std::endl;
    }
    if (dump_layers.length() > 0)
        dump_layers = " " + dump_layers + " "; // Insert delimiter for easier parsing when used
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
