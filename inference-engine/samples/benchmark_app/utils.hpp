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
bool updateShapes(InferenceEngine::ICNNNetwork::InputShapes& shapes,
                  const std::string shapes_string, const InferenceEngine::InputsDataMap& input_info);
bool adjustShapesBatch(InferenceEngine::ICNNNetwork::InputShapes& shapes,
                       const size_t batch_size, const InferenceEngine::InputsDataMap& input_info);
std::string getShapesString(const InferenceEngine::ICNNNetwork::InputShapes& shapes);

#ifdef USE_OPENCV
void dump_config(const std::string& filename,
                 const std::map<std::string, std::map<std::string, std::string>>& config);
void load_config(const std::string& filename,
                 std::map<std::string, std::map<std::string, std::string>>& config);
#endif