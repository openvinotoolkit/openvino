// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sys/stat.h>

#include <fstream>
#include <regex>
#include <string>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/w_dirent.h"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

#ifdef _WIN32
#    include <direct.h>
#    define rmdir(dir) _rmdir(dir)
#else  // _WIN32
#    include <unistd.h>
#endif  // _WIN32

namespace ov {
namespace test {
namespace utils {

template <class T>
inline std::string to_string_c_locale(T value) {
    std::stringstream val_stream;
    val_stream.imbue(std::locale("C"));
    val_stream << value;
    return val_stream.str();
}

inline std::string makePath(const std::string& folder, const std::string& file) {
    if (folder.empty())
        return file;
    return folder + FileSeparator + file;
}

inline long long fileSize(const char* fileName) {
    std::ifstream in(fileName, std::ios_base::binary | std::ios_base::ate);
    return in.tellg();
}

inline long long fileSize(const std::string& fileName) {
    return fileSize(fileName.c_str());
}

inline bool fileExists(const char* fileName) {
    return fileSize(fileName) >= 0;
}

inline bool fileExists(const std::string& fileName) {
    return fileExists(fileName.c_str());
}

inline void createFile(const std::string& filename, const std::string& content) {
    std::ofstream outfile(filename);
    outfile << content;
    outfile.close();
}

inline void removeFile(const std::string& path) {
    if (!path.empty()) {
        std::remove(path.c_str());
    }
}

inline void removeIRFiles(const std::string& xmlFilePath, const std::string& binFileName) {
    if (fileExists(xmlFilePath)) {
        std::remove(xmlFilePath.c_str());
    }

    if (fileExists(binFileName)) {
        std::remove(binFileName.c_str());
    }
}

// Removes all files with extension=ext from the given directory
// Return value:
// < 0 - error
// >= 0 - count of removed files
inline int removeFilesWithExt(std::string path, std::string ext) {
    struct dirent* ent;
    DIR* dir = opendir(path.c_str());
    int ret = 0;
    if (dir != nullptr) {
        while ((ent = readdir(dir)) != NULL) {
            auto file = makePath(path, std::string(ent->d_name));
            struct stat stat_path;
            stat(file.c_str(), &stat_path);
            if (!S_ISDIR(stat_path.st_mode) && endsWith(file, "." + ext)) {
                auto err = std::remove(file.c_str());
                if (err != 0) {
                    closedir(dir);
                    return err;
                }
                ret++;
            }
        }
        closedir(dir);
    }

    return ret;
}

// Lists all files with extension=ext from the given directory
// Return value:
// vector of strings representing file paths
inline std::vector<std::string> listFilesWithExt(const std::string& path, const std::string& ext) {
    struct dirent* ent;
    DIR* dir = opendir(path.c_str());
    std::vector<std::string> res;
    if (dir != nullptr) {
        while ((ent = readdir(dir)) != NULL) {
            auto file = makePath(path, std::string(ent->d_name));
            struct stat stat_path;
            stat(file.c_str(), &stat_path);
            if (!S_ISDIR(stat_path.st_mode) && endsWith(file, "." + ext)) {
                res.push_back(std::move(file));
            }
        }
        closedir(dir);
    }
    return res;
}

inline int removeDir(const std::string& path) {
    return rmdir(path.c_str());
}

inline bool directoryExists(const std::string& path) {
    struct stat sb;

    if (stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
        return true;
    }

    return false;
}

inline void directoryFileListRecursive(const std::string& name, std::vector<std::string>& file_list) {
    struct CloseDir {
        void operator()(DIR* d) const noexcept {
            if (d) {
                closedir(d);
            }
        }
    };
    using Dir = std::unique_ptr<DIR, CloseDir>;
    Dir directory(opendir(name.c_str()));
    struct dirent* entire;
    if (directory) {
        const std::string current_dir{"."};
        const std::string parent_dir{".."};
        while ((entire = readdir(directory.get())) != nullptr) {
            if (entire->d_name == parent_dir || entire->d_name == current_dir) {
                continue;
            }
            std::string path = name + FileSeparator + entire->d_name;
            if (directoryExists(path)) {
                directoryFileListRecursive(path, file_list);
            }
            if (fileExists(path)) {
                file_list.push_back(path);
            }
        }
    }
}

inline int createDirectory(const std::string& dirPath) {
#ifdef _WIN32
    return _mkdir(dirPath.c_str());
#else
    return mkdir(dirPath.c_str(), mode_t(0777));
#endif
}

inline int createDirectoryRecursive(const std::string& dirPath) {
    std::string copyDirPath = dirPath;
    std::vector<std::string> nested_dir_names;
    while (!directoryExists(copyDirPath)) {
        auto pos = copyDirPath.rfind(FileSeparator);
        nested_dir_names.push_back(copyDirPath.substr(pos, copyDirPath.length() - pos));
        copyDirPath = copyDirPath.substr(0, pos);
    }
    while (!nested_dir_names.empty()) {
        copyDirPath = copyDirPath + nested_dir_names.back();
        if (createDirectory(copyDirPath) != 0) {
            return -1;
        }
        nested_dir_names.pop_back();
    }
    return 0;
}

inline std::vector<std::string> getFileListByPatternRecursive(const std::vector<std::string>& folderPaths,
                                                              const std::vector<std::regex>& patterns) {
    auto getFileListByPattern = [&patterns](const std::string& folderPath) {
        std::vector<std::string> allFilePaths;
        directoryFileListRecursive(folderPath, allFilePaths);
        std::set<std::string> result;
        for (auto& filePath : allFilePaths) {
            for (const auto& pattern : patterns) {
                if (fileExists(filePath) && std::regex_match(filePath, pattern)) {
                    result.insert(filePath);
                    break;
                }
            }
        }
        return result;
    };

    std::vector<std::string> result;
    for (auto&& folderPath : folderPaths) {
        if (!directoryExists(folderPath)) {
            std::string msg = "Input directory (" + folderPath + ") doesn't not exist!";
            throw std::runtime_error(msg);
        }
        auto fileListByPattern = getFileListByPattern(folderPath);
        result.insert(result.end(), fileListByPattern.begin(), fileListByPattern.end());
    }
    return result;
}

inline std::string replaceExt(std::string file, const std::string& newExt) {
    std::string::size_type i = file.rfind('.', file.length());

    if (i != std::string::npos) {
        if (newExt == "") {
            file = file.substr(0, i);
        } else {
            file.replace(i + 1, newExt.length(), newExt);
        }
    }
    return file;
}

inline std::vector<std::string> splitStringByDelimiter(std::string paths, const std::string& delimiter = ",") {
    size_t delimiterPos;
    std::vector<std::string> splitPath;
    while ((delimiterPos = paths.find(delimiter)) != std::string::npos) {
        splitPath.push_back(paths.substr(0, delimiterPos));
        paths = paths.substr(delimiterPos + 1);
    }
    splitPath.push_back(paths);
    return splitPath;
}

std::string getModelFromTestModelZoo(const std::string& relModelPath);

inline std::vector<std::string> readListFiles(const std::vector<std::string>& filePaths) {
    std::vector<std::string> res;
    for (const auto& filePath : filePaths) {
        if (!fileExists(filePath)) {
            std::string msg = "Input directory (" + filePath + ") doesn't not exist!";
            throw std::runtime_error(msg);
        }
        std::ifstream file(filePath);
        if (file.is_open()) {
            std::string buffer;
            while (getline(file, buffer)) {
                if (buffer.find("#") == std::string::npos && !buffer.empty()) {
                    res.emplace_back(buffer);
                }
            }
        } else {
            std::string msg = "Error in opening file: " + filePath;
            throw std::runtime_error(msg);
        }
        file.close();
    }
    return res;
}

std::string getExecutableDirectory();
std::string getCurrentWorkingDir();
std::string getRelativePath(const std::string& from, const std::string& to);

namespace {
inline std::string get_mock_engine_path() {
    std::string mockEngineName("mock_engine");
    return ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                              mockEngineName + OV_BUILD_POSTFIX);
}

template <class T>
std::function<T> make_std_function(const std::shared_ptr<void> so, const std::string& functionName) {
    std::function<T> ptr(reinterpret_cast<T*>(ov::util::get_symbol(so, functionName.c_str())));
    return ptr;
}

}  // namespace

class MockPlugin : public ov::IPlugin {
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    void set_property(const ov::AnyMap& properties) override {
        for (auto&& it : properties) {
            if (it.first == ov::num_streams.name())
                num_streams = it.second.as<ov::streams::Num>();
        }
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
        if (name == ov::supported_properties) {
            std::vector<ov::PropertyName> supportedProperties = {
                ov::PropertyName(ov::supported_properties.name(), ov::PropertyMutability::RO),
                ov::PropertyName(ov::num_streams.name(), ov::PropertyMutability::RW)};
            return decltype(ov::supported_properties)::value_type(supportedProperties);
        } else if (name == ov::internal::supported_properties) {
            return decltype(ov::internal::supported_properties)::value_type({});
        } else if (name == ov::num_streams.name()) {
            return decltype(ov::num_streams)::value_type(num_streams);
        }
        return "";
    }

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

private:
    int32_t num_streams{0};
};

}  // namespace utils
}  // namespace test
}  // namespace ov
