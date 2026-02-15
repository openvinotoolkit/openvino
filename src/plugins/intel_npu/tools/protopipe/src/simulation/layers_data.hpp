//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>

#include "scenario/inference.hpp"
#include "utils/data_providers.hpp"

std::string normalizeLayerName(const std::string& layer_name);
std::vector<cv::Mat> uploadLayerData(const std::filesystem::path& path, const std::string& tag, const LayerInfo& layer);

enum class LayersType { INPUT = 0, OUTPUT };
using LayersDataMap = std::unordered_map<std::string, std::vector<cv::Mat>>;
LayersDataMap uploadFromDirectory(const std::filesystem::path& path, const std::string& tag, const LayersInfo& layers);

LayersDataMap uploadData(const std::filesystem::path& path, const std::string& tag, const LayersInfo& layers,
                         LayersType type);

bool isDirectory(const std::filesystem::path& path);

std::vector<IDataProvider::Ptr> createConstantProviders(LayersDataMap&& layers_data,
                                                        const std::vector<std::string>& layer_names);

std::vector<IDataProvider::Ptr> createRandomProviders(const LayersInfo& layers,
                                                      const std::map<std::string, IRandomGenerator::Ptr>& generators);

std::vector<std::filesystem::path> createDirectoryLayout(const std::filesystem::path& path,
                                                         const std::vector<std::string>& layer_names);
template <typename T>
std::map<std::string, T> unpackWithDefault(const LayerVariantAttr<T>& attr, const std::vector<std::string>& layer_names,
                                           const T& def_value) {
    std::map<std::string, T> result;
    if (std::holds_alternative<std::monostate>(attr)) {
        for (const auto& layer_name : layer_names) {
            result.emplace(layer_name, def_value);
        }
    } else if (std::holds_alternative<T>(attr)) {
        auto val = std::get<T>(attr);
        for (const auto& layer_name : layer_names) {
            result.emplace(layer_name, val);
        }
    } else {
        auto map = std::get<AttrMap<T>>(attr);
        for (const auto& layer_name : layer_names) {
            if (auto it = map.find(layer_name); it != map.end()) {
                result.emplace(layer_name, it->second);
            } else {
                result.emplace(layer_name, def_value);
            }
        }
    }
    return result;
}
