// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <regex>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <sys/stat.h>

#include "test_constants.hpp"
#include "w_dirent.h"
#include "common_utils.hpp"

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#define rmdir(dir) _rmdir(dir)
#else  // _WIN32
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif  // _WIN32

namespace CommonTestUtils {

template<class T>
inline std::string to_string_c_locale(T value) {
    std::stringstream val_stream;
    val_stream.imbue(std::locale("C"));
    val_stream << value;
    return val_stream.str();
}

inline std::string makePath(const std::string &folder, const std::string &file) {
    if (folder.empty()) return file;
    return folder + FileSeparator + file;
}

inline long long fileSize(const char *fileName) {
    std::ifstream in(fileName, std::ios_base::binary | std::ios_base::ate);
    return in.tellg();
}

inline long long fileSize(const std::string &fileName) {
    return fileSize(fileName.c_str());
}

inline bool fileExists(const char *fileName) {
    return fileSize(fileName) >= 0;
}

inline bool fileExists(const std::string &fileName) {
    return fileExists(fileName.c_str());
}

inline void createFile(const std::string& filename, const std::string& content) {
    std::ofstream outfile(filename);
    outfile << content;
    outfile.close();
}

inline void removeFile(const std::string& path) {
    if (!path.empty()) {
        remove(path.c_str());
    }
}

inline void removeIRFiles(const std::string &xmlFilePath, const std::string &binFileName) {
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
    struct dirent *ent;
    DIR *dir = opendir(path.c_str());
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
    struct dirent *ent;
    DIR *dir = opendir(path.c_str());
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

inline int removeDir(const std::string &path) {
    return rmdir(path.c_str());
}

inline bool directoryExists(const std::string &path) {
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
    struct dirent *entire;
    if (directory) {
        const std::string current_dir{"."};
        const std::string parent_dir{".."};
        while ((entire = readdir(directory.get())) != nullptr) {
            if (entire->d_name == parent_dir || entire->d_name == current_dir) {
                continue;
            }
            std::string path = name + CommonTestUtils::FileSeparator + entire->d_name;
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
        auto pos = copyDirPath.rfind(CommonTestUtils::FileSeparator);
        nested_dir_names.push_back(copyDirPath.substr(pos, copyDirPath.length() - pos));
        copyDirPath = copyDirPath.substr(0, pos);
    }
    while (!nested_dir_names.empty()) {
        std::string a = copyDirPath + nested_dir_names.back();
        if (createDirectory(a) != 0) {
            return -1;
        }
        nested_dir_names.pop_back();
    }
    return 0;
}

inline std::vector<std::string> getFileListByPatternRecursive(const std::vector<std::string>& folderPaths,
                                                              const std::regex& pattern) {
    auto getFileListByPattern = [&pattern](const std::string& folderPath) {
        std::vector<std::string> allFilePaths;
        CommonTestUtils::directoryFileListRecursive(folderPath, allFilePaths);
        std::set<std::string> result;
        for (auto& filePath : allFilePaths) {
            if (CommonTestUtils::fileExists(filePath) && std::regex_match(filePath, pattern)) {
                result.insert(filePath);
            }
        }
        return result;
    };

    std::vector<std::string> result;
    for (auto &&folderPath : folderPaths) {
        if (!CommonTestUtils::directoryExists(folderPath)) {
            continue;
        }
        auto fileListByPattern = getFileListByPattern(folderPath);
        result.insert(result.end(), fileListByPattern.begin(), fileListByPattern.end());
    }
    return result;
}

inline std::string replaceExt(std::string file, const std::string& newExt) {
    std::string::size_type i = file.rfind('.', file.length());

    if (i != std::string::npos) {
        file.replace(i + 1, newExt.length(), newExt);
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

inline void lockAndWaitFile(const std::string& lockedFilename) {
    unsigned int processId = getpid();
    if (CommonTestUtils::fileExists(lockedFilename.c_str())) {
        std::ifstream lockedFile;
        lockedFile.open(lockedFilename);
        std::string content;
        lockedFile >> content;

        auto updateLockedFile = [&lockedFile, &lockedFilename, &processId]() {
            lockedFile.clear();
            std::ofstream ostream(lockedFilename);
            ostream << processId;
            ostream.close();
        };

        unsigned int lockerPid;
        try {
            lockerPid = std::stoi(content);
        } catch (...) {
            updateLockedFile();
            lockedFile.close();
            return;
        }

        if (lockerPid != processId) {
            auto currentTime = std::chrono::system_clock::now();
            auto exitTime = currentTime + std::chrono::minutes(1);
            unsigned int timeout = 1e6 * 0.5;
            while (CommonTestUtils::fileExists(lockedFilename.c_str()) || currentTime >= exitTime) {
                currentTime += std::chrono::microseconds(timeout);
                std::this_thread::sleep_for(std::chrono::microseconds(timeout));
            }
            int status = 0;
#ifndef _WIN32
            waitpid(lockerPid, &status, WNOHANG);
#else
            DWORD winProcessId = static_cast<DWORD>(processId);
            HANDLE winProcess = ::OpenProcess(PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION |
                                       PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ,
                                       FALSE, winProcessId);
            if (winProcess != NULL) {
               LPDWORD winStatus = NULL;
                ::WaitForSingleObject(winProcess, INFINITE);
                ::GetExitCodeProcess(winProcess, winStatus);
                ::CloseHandle(winProcess);
                status = static_cast<int>(*winStatus);
            }
#endif
            if (status != 0 && currentTime >= exitTime) {
                updateLockedFile();
            }
        }
        lockedFile.close();
    } else {
        std::string content = std::to_string(processId);
        CommonTestUtils::createFile(lockedFilename, content);
    }
}
}  // namespace CommonTestUtils
