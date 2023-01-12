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

ov::util::FilePath ov::getPluginPath(const std::string& pluginName) {
    // Assume pluginName may contain:
    // 1. /path/to/libexample.so absolute path
    // 2. ../path/to/libexample.so path relative to working directory
    // 3. libexample.so path relative to working directory
    // 4. example library name - will be searched in library directory

    // For 4th case - try to find in library directory
    if (!ov::util::ends_with(pluginName, ov::util::FileTraits<char>::library_ext())) {
        auto libName = FileUtils::makePluginLibraryName({}, pluginName);
        // TODO: search as 3rd case, don't force user to put libs in library directory
        return getFilePathFromLibDir(libName);
    }

    // For 1st-3rd cases - make path absolute
    return ov::util::to_file_path(ov::util::get_absolute_file_path(pluginName));
}

ov::util::FilePath ov::getPluginPathFromXML(const std::string& pluginPath) {
    // Assume plugins.xml "location" record contains only:
    // 1. /path/to/libexample.so absolute path
    // 2. ../path/to/libexample.so path relative to library directory
    // 3. libexample.so path relative to library directory

    if (ov::util::is_absolute_file_path(pluginPath))
        return ov::util::to_file_path(pluginPath);

    // TODO: for cases 2-3 search relative to XML file instead of LibDir
    return ov::getFilePathFromLibDir(pluginPath);
}