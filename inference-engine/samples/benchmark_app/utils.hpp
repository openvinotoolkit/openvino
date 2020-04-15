// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <map>

std::vector<std::string> parseDevices(const std::string& device_string);
uint32_t deviceDefaultDeviceDurationInSeconds(const std::string& device);
std::map<std::string, std::string> parseNStreamsValuePerDevice(const std::vector<std::string>& devices,
                                                               const std::string& values_string);
#ifdef USE_OPENCV
void dump_config(const std::string& filename,
                 const std::map<std::string, std::map<std::string, std::string>>& config);
void load_config(const std::string& filename,
                 std::map<std::string, std::map<std::string, std::string>>& config);
#endif