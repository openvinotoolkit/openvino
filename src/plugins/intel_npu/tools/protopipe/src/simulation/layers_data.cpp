//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layers_data.hpp"

#include <fstream>
#include <regex>

#include "utils/error.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"

std::string normalizeLayerName(const std::string& layer_name) {
    std::string normalized = layer_name;
    std::unordered_set<char> prohibited = {'\\', '/', ':', '*', '?', '"', '<', '>'};
    std::replace_if(
            normalized.begin(), normalized.end(),
            [&prohibited](char ch) {
                return prohibited.find(ch) != prohibited.end();
            },
            '_');
    return normalized;
};

std::vector<cv::Mat> uploadLayerData(const std::filesystem::path& path, const std::string& tag,
                                     const LayerInfo& layer) {
    if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path)) {
        THROW_ERROR("Failed to find data folder: " << path << " for model: " << tag << ", layer: " << layer.name);
    }
    std::string iter_file_pattern = "iter_(\\d+)\\.bin";
    std::regex regex(iter_file_pattern);
    std::unordered_map<int, std::filesystem::path> iter_files_map;
    for (const auto& entry : std::filesystem::directory_iterator{path}) {
        std::smatch match;
        const auto& filename = entry.path().filename().string();
        if (std::regex_match(filename, match, regex)) {
            const auto iter_idx = std::stoi(match[1].str());
            iter_files_map.emplace(iter_idx, entry);
        }
    }
    std::vector<cv::Mat> out_mats;
    for (int i = 0; i < iter_files_map.size(); ++i) {
        if (auto it = iter_files_map.find(i); it != iter_files_map.end()) {
            cv::Mat mat;
            utils::createNDMat(mat, layer.dims, layer.prec);
            utils::readFromBinFile(it->second.string(), mat);
            out_mats.push_back(std::move(mat));
        } else {
            THROW_ERROR("Failed to find data for iteration: " << i << ", model: " << tag << ", layer: " << layer.name);
        }
    }
    return out_mats;
}

using LayersDataMap = std::unordered_map<std::string, std::vector<cv::Mat>>;
LayersDataMap uploadFromDirectory(const std::filesystem::path& path, const std::string& tag, const LayersInfo& layers) {
    LayersDataMap layers_data;
    for (const auto& layer : layers) {
        auto normalized = normalizeLayerName(layer.name);
        auto data = uploadLayerData(path / normalized, tag, layer);
        if (data.empty()) {
            THROW_ERROR("No iterations data found for model: " << tag << ", layer: " << layer.name);
        }
        LOG_INFO() << "    - Found " << data.size() << " iteration(s) for layer: " << layer.name << std::endl;
        layers_data.emplace(layer.name, std::move(data));
    }
    return layers_data;
}

LayersDataMap uploadData(const std::filesystem::path& path, const std::string& tag, const LayersInfo& layers,
                         LayersType type) {
    ASSERT(!layers.empty());
    const std::string kLayersTypeStr = type == LayersType::INPUT ? "input" : "output";
    if (!std::filesystem::exists(path)) {
        THROW_ERROR("" << path << " must exist to upload layers data!")
    }
    LayersDataMap layers_data;
    if (std::filesystem::is_directory(path)) {
        layers_data = uploadFromDirectory(path, tag, layers);
    } else {
        if (layers.size() > 1u) {
            THROW_ERROR("Model: " << tag << " must have exactly one " << kLayersTypeStr
                                  << " layer in order to upload data from: " << path);
        }
        const auto& layer = layers.front();
        cv::Mat mat;
        utils::createNDMat(mat, layer.dims, layer.prec);
        utils::readFromBinFile(path.string(), mat);
        LOG_INFO() << "    - Found single iteration data for model: " << tag << ", layer: " << layer.name << std::endl;
        layers_data = {{layer.name, std::vector<cv::Mat>{mat}}};
    }
    // NB: layers_data can't be empty as long as layers vector is non-empty.
    const auto kNumPerLayerIterations = layers_data.begin()->second.size();
    // NB: All i/o layers for model must have the equal amount of data.
    for (const auto& [layer_name, data_vec] : layers_data) {
        if (data_vec.size() != kNumPerLayerIterations) {
            THROW_ERROR("Model: " << tag << " has different amount of data for " << kLayersTypeStr
                                  << " layer: " << layer_name << "(" << data_vec.size() << ") and layer: "
                                  << layers_data.begin()->first << "(" << kNumPerLayerIterations << ")");
        }
    }
    return layers_data;
}

bool isDirectory(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        return std::filesystem::is_directory(path);
    }
    return path.extension().empty();
}

std::vector<IDataProvider::Ptr> createConstantProviders(LayersDataMap&& layers_data,
                                                        const std::vector<std::string>& layer_names) {
    std::vector<IDataProvider::Ptr> providers;
    for (const auto& layer_name : layer_names) {
        auto layer_data = layers_data.at(layer_name);
        providers.push_back(std::make_shared<CircleBuffer>(std::move(layer_data)));
    }
    return providers;
}

std::vector<IDataProvider::Ptr> createRandomProviders(const LayersInfo& layers,
                                                      const std::map<std::string, IRandomGenerator::Ptr>& generators) {
    std::vector<IDataProvider::Ptr> providers;
    for (const auto& layer : layers) {
        auto generator = generators.at(layer.name);
        auto provider = std::make_shared<RandomProvider>(generator, layer.dims, layer.prec);
        LOG_INFO() << "    - Random generator: " << generator->str() << " will be used for layer: " << layer.name
                   << std::endl;
        providers.push_back(std::move(provider));
    }
    return providers;
}

std::vector<std::filesystem::path> createDirectoryLayout(const std::filesystem::path& path,
                                                         const std::vector<std::string>& layer_names) {
    std::vector<std::filesystem::path> dirs_path;
    std::filesystem::create_directories(path);
    for (const auto& layer_name : layer_names) {
        // NB: Use normalized layer name to create dir
        // to store reference data for particular layer.
        std::filesystem::path curr_dir = path / normalizeLayerName(layer_name);
        dirs_path.push_back(curr_dir);
        std::filesystem::create_directory(curr_dir);
        {
            // NB: Save the original layer name;
            std::ofstream file{curr_dir / "layer_name.txt"};
            ASSERT(file.is_open());
            file << layer_name;
        }
    }
    return dirs_path;
}
