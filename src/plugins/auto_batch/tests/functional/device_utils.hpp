// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <string>
namespace ov {
namespace device_utils {

bool is_device_supported(std::string name);
bool is_gpu_device_supported();
bool is_target_device_disabled(std::string name, std::string target);

}  // namespace device_utils
}  // namespace ov
