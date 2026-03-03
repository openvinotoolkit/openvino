// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sys/stat.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <variant>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
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

namespace ov::test::utils {

/// OS specific file traits
template <class C>
struct FileTraits;

template <>
struct FileTraits<char> {
    static constexpr const auto file_separator =
#ifdef _WIN32
        '\\';
#else
        '/';
#endif
    static constexpr const auto dot_symbol = '.';
    static std::string library_ext() {
#ifdef _WIN32
        return {"dll"};
#else
        return {"so"};
#endif
    }
    static std::string library_prefix() {
#ifdef _WIN32
#    if defined(__MINGW32__) || defined(__MINGW64__)
        return {"lib"};
#    else
        return {""};
#    endif
#else
        return {"lib"};
#endif
    }
};

template <>
struct FileTraits<wchar_t> {
    static constexpr const auto file_separator =
#ifdef _WIN32
        L'\\';
#else
        L'/';
#endif
    static constexpr const auto dot_symbol = L'.';
    static std::wstring library_ext() {
#ifdef _WIN32
        return {L"dll"};
#else
        return {L"so"};
#endif
    }
    static std::wstring library_prefix() {
#ifdef _WIN32
#    if defined(__MINGW32__) || defined(__MINGW64__)
        return {L"lib"};
#    else
        return {L""};
#    endif
#else
        return {L"lib"};
#endif
    }
};

template <class T>
inline std::string to_string_c_locale(T value) {
    std::stringstream val_stream;
    val_stream.imbue(std::locale("C"));
    val_stream << value;
    return val_stream.str();
}

template <class C, std::enable_if_t<(std::is_same_v<C, char> || std::is_same_v<C, wchar_t>)>* = nullptr>
inline std::basic_string<C> makePath(const std::basic_string<C>& folder, const std::basic_string<C>& file) {
    return folder.empty() ? file : folder + FileTraits<C>::file_separator + file;
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
        std::filesystem::remove(ov::util::make_path(xmlFilePath));
    }

    if (fileExists(binFileName)) {
        std::filesystem::remove(ov::util::make_path(binFileName));
    }
}

// Removes all files with extension=ext from the given directory
// Return value:
// < 0 - error
// >= 0 - count of removed files
template <bool Force = !opt::FORCE>
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
                if constexpr (Force) {
                    std::filesystem::permissions(file,
                                                 std::filesystem::perms::owner_write,
                                                 std::filesystem::perm_options::add);
                }
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

inline int createDirectory(const std::string& dirPath) {
#ifdef _WIN32
    return _mkdir(dirPath.c_str());
#else
    return mkdir(dirPath.c_str(), mode_t(0777));
#endif
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
inline int createDirectory(const std::wstring& dirPath) {
#    ifdef _WIN32
    return _wmkdir(dirPath.c_str());
#    else
    return mkdir(ov::util::wstring_to_string(dirPath).c_str(), mode_t(0777));
#    endif
}
#endif

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

std::string getOpenvinoLibDirectory();
std::string getExecutableDirectory();
std::string getCurrentWorkingDir();
std::string getRelativePath(const std::string& from, const std::string& to);

namespace {
inline std::filesystem::path get_mock_engine_path() {
    std::string mockEngineName("mock_engine");
    return ov::util::make_plugin_library_name(ov::util::make_path(ov::test::utils::getExecutableDirectory()),
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

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::AnyMap& properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
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

using StringPathVariant = std::variant<std::string, std::u16string, std::u32string, std::wstring>;

std::filesystem::path to_fs_path(const StringPathVariant& param);

}  // namespace ov::test::utils
