// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_core_get_plugin.hpp"

#include "file_utils.h"
#include "openvino/core/core.hpp"
#include "openvino/util/common_util.hpp"

ov::util::FilePath ov::getFilePathFromLibDir(const std::string& filePath) {
    // Assume filePath contains only path relative to library directory
    // (with libopenvino.so)
    const auto ieLibraryPath = InferenceEngine::getInferenceEngineLibraryPath();

    auto filePath_ = ov::util::to_file_path(filePath);

    // file can be found either:

    // 1. in openvino-X.Y.Z folder relative to libopenvino.so
    std::ostringstream str;
    str << "openvino-" << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "." << OPENVINO_VERSION_PATCH;
    const auto subFolder = ov::util::to_file_path(str.str());

    ov::util::FilePath absFilePath = FileUtils::makePath(FileUtils::makePath(ieLibraryPath, subFolder), filePath_);
    if (FileUtils::fileExist(absFilePath))
        return absFilePath;

    // 2. in the libopenvino.so location
    absFilePath = FileUtils::makePath(ieLibraryPath, filePath_);
    return absFilePath;
}

ov::util::FilePath ov::getPluginPath(const std::string& plugin) {
    // Assume `plugin` may contain:
    // 1. /path/to/libexample.so absolute path
    // 2. ../path/to/libexample.so path relative to working directory
    // 3. example library name - to be converted to 4th case
    // 4. libexample.so - path relative to working directory (if exists) or file to be found in ENV

    // For 1-2 cases
    if (plugin.find(ov::util::FileTraits<char>::file_separator) != std::string::npos)
        return ov::util::to_file_path(ov::util::get_absolute_file_path(plugin));

    auto libName = plugin;
    // For 3rd case - convert to 4th case
    if (!ov::util::ends_with(plugin, ov::util::FileTraits<char>::library_ext()))
        libName = FileUtils::makePluginLibraryName({}, plugin);

    // For 4th case
    auto libPath = ov::util::to_file_path(ov::util::get_absolute_file_path(libName));
    if (ov::util::file_exists(libPath))
        return libPath;
    return ov::util::to_file_path(libName);
}

ov::util::FilePath ov::getPluginPath(const std::string& plugin, const std::string& xmlPath, bool asAbsOnly) {
    // Assume `plugin` (from XML "location" record) contains only:
    // 1. /path/to/libexample.so absolute path
    // 2. ../path/to/libexample.so path relative to XML directory
    // 3. example library name - to be converted to 4th case
    // 4. libexample.so - path relative to XML directory (if exists) or file to be found in ENV (if `load_as_abs_only is False`)

    // For 1st case
    if (ov::util::is_absolute_file_path(plugin))
        return ov::util::to_file_path(plugin);

    auto xmlPath_ = xmlPath;
    if (xmlPath.find(ov::util::FileTraits<char>::file_separator) == std::string::npos)
        xmlPath_ = FileUtils::makePath(std::string("."), xmlPath);    // treat plugins.xml as CWD/plugins.xml

    // For 2nd case
    if (plugin.find(ov::util::FileTraits<char>::file_separator) != std::string::npos) {
        auto path_ = FileUtils::makePath(ov::util::get_directory(xmlPath_), plugin);
        return ov::util::to_file_path(ov::util::get_absolute_file_path(path_));  // canonicalize path
    }

    auto libFileName = plugin;
    // For 3rd case - convert to 4th case
    if (!ov::util::ends_with(plugin, ov::util::FileTraits<char>::library_ext()))
        libFileName = FileUtils::makePluginLibraryName({}, plugin);

    // For 4th case
    auto libPath = FileUtils::makePath(ov::util::get_directory(xmlPath_), libFileName);
    libPath = ov::util::get_absolute_file_path(libPath);  // canonicalize path
    if (asAbsOnly || ov::util::file_exists(libPath))
        return ov::util::to_file_path(libPath);
    return ov::util::to_file_path(libFileName);
}