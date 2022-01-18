// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Basic function to work with file system and UNICODE symbols
 * @file file_utils.h
 */

#pragma once

// clang-format off
#include <string>
#include <cstring>

#include "ie_api.h"
#include "openvino/util/file_util.hpp"

/// @ingroup ie_dev_api_file_utils
namespace FileUtils {

/**
 * @brief Enables only `char` or `wchar_t` template specializations
 * @tparam C A char type
 */
template <typename C>
using enableIfSupportedChar =
    typename std::enable_if<(std::is_same<C, char>::value || std::is_same<C, wchar_t>::value)>::type;

/**
 * @brief Interface function to get absolute path of file
 * @ingroup ie_dev_api_file_utils
 * @param filePath - path to file, can be relative to current working directory
 * @return Absolute path of file
 * @throw InferenceEngine::Exception if any error occurred
 */
INFERENCE_ENGINE_API_CPP(std::string) absoluteFilePath(const std::string& filePath);

/**
 * @brief Interface function to create directorty recursively by given path
 * @ingroup ie_dev_api_file_utils
 * @param dirPath - path to file, can be relative to current working directory
 * @throw InferenceEngine::Exception if any error occurred
 */
INFERENCE_ENGINE_API_CPP(void) createDirectoryRecursive(const std::string& dirPath);

/**
 * @brief Interface function to check if directory exists for given path
 * @ingroup ie_dev_api_file_utils
 * @param path - path to directory
 * @return true if directory exists, false otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) directoryExists(const std::string& path);

/**
 * @brief Interface function to get the size of a file. The function supports UNICODE path
 * @ingroup ie_dev_api_file_utils
 * @param fileName - name of the file
 * @return size of the file
 */
INFERENCE_ENGINE_API(long long) fileSize(const char *fileName);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief      Returns file size for file with UNICODE path name
 * @ingroup    ie_dev_api_file_utils
 *
 * @param[in]  fileName  The file name
 *
 * @return     { description_of_the_return_value }
 */
inline long long fileSize(const wchar_t* fileName) {
    return fileSize(::ov::util::wstring_to_string(fileName).c_str());
}

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief Function to get the size of a file. The function supports UNICODE path
 * @ingroup ie_dev_api_file_utils
 * @param f - string name of the file
 * @return size of the file
 */
template <typename C, typename = enableIfSupportedChar<C>>
inline long long fileSize(const std::basic_string<C> &f) {
    return fileSize(f.c_str());
}

/**
 * @brief check if file with a given filename exists. The function supports UNICODE path
 * @ingroup ie_dev_api_file_utils
 * @param fileName - given filename
 * @return true is exists
 */
template <typename C, typename = enableIfSupportedChar<C>>
inline bool fileExist(const C * fileName) {
    return fileSize(fileName) >= 0;
}

/**
 * @brief check if file with a given filename exists.  The function supports UNICODE path
 * @ingroup ie_dev_api_file_utils
 * @param fileName - string with a given filename
 * @return true is exists
 */
template <typename C, typename = enableIfSupportedChar<C>>
inline bool fileExist(const std::basic_string<C> &fileName) {
    return fileExist(fileName.c_str());
}

/**
 * @brief CPP Interface function to combint path with filename. The function supports UNICODE path
 * @ingroup ie_dev_api_file_utils
 * @param folder - path to add filename to
 * @param file - filename to add to path
 * @return string with combination of the path and the filename divided by file separator
 */

template <typename C, typename = enableIfSupportedChar<C>>
inline std::basic_string<C> makePath(const std::basic_string<C> &folder, const std::basic_string<C> &file) {
    if (folder.empty())
        return file;
    return folder + ov::util::FileTraits<C>::file_separator + file;
}

/**
 * @brief CPP Interface function to extract extension from filename
 * @ingroup ie_dev_api_file_utils
 * @param filename - string with the name of the file which extension should be extracted
 * @return string with extracted file extension
 */
template <typename C, typename = enableIfSupportedChar<C>>
inline std::basic_string<C> fileExt(const std::basic_string<C> &filename) {
    auto pos = filename.rfind(ov::util::FileTraits<C>::dot_symbol);
    if (pos == std::string::npos)
        return {};
    return filename.substr(pos + 1);
}

template <typename C, typename = enableIfSupportedChar<C>>
inline std::basic_string<C> makePluginLibraryName(const std::basic_string<C> &path, const std::basic_string<C> &input) {
    std::basic_string<C> separator(1, ov::util::FileTraits<C>::file_separator);
    if (path.empty())
        separator = {};
    return path + separator + ov::util::FileTraits<C>::library_prefix() + input + ov::util::FileTraits<C>::dot_symbol + ov::util::FileTraits<C>::library_ext();
}

}  // namespace FileUtils
// clang-format on

namespace InferenceEngine {

/**
 * @brief   Returns a path to Inference Engine library
 * @ingroup ie_dev_api_file_utils
 * @return  A `std::string` path to Inference Engine library
 */
INFERENCE_ENGINE_API_CPP(std::string) getIELibraryPath();

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief   Returns a unicode path to Inference Engine library
 * @ingroup ie_dev_api_file_utils
 * @return  A `std::wstring` path to Inference Engine library
 */
INFERENCE_ENGINE_API_CPP(std::wstring) getIELibraryPathW();

inline ::ov::util::FilePath getInferenceEngineLibraryPath() {
    return getIELibraryPathW();
}

#else

inline ::ov::util::FilePath getInferenceEngineLibraryPath() {
    return getIELibraryPath();
}

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

}  // namespace InferenceEngine
