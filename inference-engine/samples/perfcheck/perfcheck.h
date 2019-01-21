// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

static constexpr std::size_t MIN_ITERATIONS = 1000;
static constexpr std::size_t MAX_NETWORKS   = 16;

/// @brief message for model argument
static constexpr char model_message[] = "Required. Path to an .xml file with a trained model.";
DEFINE_string(m, "", model_message);

/// @brief message for help argument
static constexpr char help_message[] = "Optional. Print a usage message.";
DEFINE_bool(h, false, help_message);

/// @brief message target_device argument
static constexpr char target_device_message[] = "Optional. Specify the target device to infer on. " \
"Sample will look for a suitable plugin for device specified. Default: CPU.";
DEFINE_string(d, "CPU", target_device_message);

/// @brief message for plugin_path argument
static constexpr char plugin_path_message[] = "Optional. Path to a plugin folder.";
DEFINE_string(pp, "", plugin_path_message);

/// @brief message for custom_cpu_library argument
static constexpr char custom_cpu_library_message[] = "Optional. Required for CPU custom layers. " \
"Absolute path to a shared library with the kernels implementation.";
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief message for custom_gpu_library argument
static constexpr char custom_gpu_library_message[] = "Optional. Required for GPU custom kernels. "\
"Absolute path to the xml file with the kernels description.";
DEFINE_string(c, "",  custom_gpu_library_message);

/// @brief message for inputs_dir argument
static constexpr char inputs_dir_message[] = "Optional. Path to a folder with images and binaries for inputs. " \
"Default value: \".\".";
DEFINE_string(inputs_dir, ".", inputs_dir_message);

/// @brief message for config argument
static constexpr char config_message[] = "Optional. Path to a configuration file.";
DEFINE_string(config, "", config_message);

/// @brief message for num_iterations argument
static constexpr char num_iterations_message[] = "Optional. Specify number of iterations. " \
"Default value: 1000. Must be greater than or equal to 1000.";
DEFINE_uint32(num_iterations, MIN_ITERATIONS, num_iterations_message);

/// @brief message for batch argument
static constexpr char batch_message[] = "Optional. Specify batch. Default value: 1.";
DEFINE_uint32(batch, 1, batch_message);

/// @brief message for num_networks argument
static constexpr char num_networks_message[] = "Optional. Specify number of networks. Default value: 1. Must be less than or equal to 16";
DEFINE_uint32(num_networks, 1, num_networks_message);

/// @brief message for num_requests argument
static constexpr char num_requests_message[] = "Optional. Specify number of infer requests. " \
"Default value depends on specified device.";
DEFINE_uint32(num_requests, 0, num_requests_message);

/// @brief message for num_fpga_devices argument
static constexpr char num_fpga_devices_message[]  = "Optional. Specify number of FPGA devices. Default value: 1.";
DEFINE_uint32(num_fpga_devices, 1, num_fpga_devices_message);

/**
* \brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "perfcheck [OPTIONS]" << std::endl;
    std::cout << "[OPTIONS]:" << std::endl;
    std::cout << "\t-m                \t <value> \t" << model_message              << std::endl;
    std::cout << "\t-h                \t         \t" << help_message               << std::endl;
    std::cout << "\t-d                \t <value> \t" << target_device_message      << std::endl;
    std::cout << "\t-pp               \t <value> \t" << plugin_path_message        << std::endl;
    std::cout << "\t-l                \t <value> \t" << custom_cpu_library_message << std::endl;
    std::cout << "\t-c                \t <value> \t" << custom_gpu_library_message << std::endl;
    std::cout << "\t-inputs_dir       \t <value> \t" << inputs_dir_message         << std::endl;
    std::cout << "\t-config           \t <value> \t" << config_message             << std::endl;
    std::cout << "\t-num_iterations   \t <value> \t" << num_iterations_message     << std::endl;
    std::cout << "\t-batch            \t <value> \t" << batch_message              << std::endl;
    std::cout << "\t-num_networks     \t <value> \t" << num_networks_message       << std::endl;
    std::cout << "\t-num_requests     \t <value> \t" << num_requests_message       << std::endl;
    std::cout << "\t-num_fpga_devices \t <value> \t" << num_fpga_devices_message   << std::endl;

    std::cout << std::endl;
}
