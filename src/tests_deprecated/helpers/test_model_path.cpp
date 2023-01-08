// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Created by user on 19.10.18.
//

#include "test_model_repo.hpp"
#include "test_model_path.hpp"

#ifndef _WIN32
# include <libgen.h>
# include <dirent.h>
#else
# include <samples/os/windows/w_dirent.h>
#endif

#include <utility>
#include <string>
#include <vector>
#include <iostream>
#include <gtest/gtest.h>
#include <fstream>

#ifndef _WIN32
static std::string getDirname(std::string filePath) {
    std::vector<char> input(filePath.begin(), filePath.end());
    input.push_back(0);
    return dirname(&*input.begin());
}
#else
static std::string getDirname(std::string filePath) {
    char dirname[_MAX_DIR] = {};
    char drive[_MAX_DRIVE] = {};
    _splitpath(filePath.c_str(), drive, dirname, nullptr, nullptr);  // preserve drive letter if possible
    return std::string(drive) + dirname;
}
#endif

static std::string get_models_path() {
    const char* models_path = TestDataHelpers::getModelPathNonFatal();

    if (nullptr == models_path) {
        ::testing::AssertionFailure() << "MODELS_PATH not defined";
    }

    return std::string(models_path);
}

static bool exist(const std::string& name) {
    std::ifstream file(name);
    if(!file)            // If the file was not found, then file is 0, i.e. !file=1 or true.
        return false;    // The file was not found.
    else                 // If the file was found, then file is non-0.
        return true;     // The file was found.
}

static std::vector<std::string> getModelsDirs() {
    auto repo_list = get_model_repo();
    int last_delimiter = 0;
    std::vector<std::string> folders;
    for(;;) {
        auto folderDelimiter = repo_list.find(':', last_delimiter);
        if (folderDelimiter == std::string::npos) {
            break;
        }
        auto nextDelimiter = repo_list.find(';', last_delimiter);
        folders.push_back(repo_list.substr(last_delimiter, folderDelimiter - last_delimiter));

        if (nextDelimiter == std::string::npos) {
            break;
        }

        last_delimiter = nextDelimiter + 1;
    }
    return folders;
}

ModelsPath::operator std::string() const {
    std::vector<std::string> absModelsPath;
    for (auto & path  : getModelsDirs()) {
        std::string b = get_models_path();
        const auto absPath = get_models_path() + kPathSeparator + path + _rel_path.str();
        absModelsPath.push_back(absPath);
        if (exist(absPath)) {
            return absPath;
        }
        //checking models for precision encoded in folder name
        auto dirname = getDirname(absPath);
        std::vector<std::pair<std::string, std::string>> stdprecisions = {
            {"_fp32", "FP32"},
            {"_q78", "_Q78"},
            {"_fp16", "FP16"},
            {"_i16", "I16"}
        };

        if (dirname.size() == absPath.size())
            continue;

        auto filename = absPath.substr(dirname.size() + 1);

        for (auto &precision : stdprecisions) {
            auto havePrecision = filename.find(precision.first);
            if (havePrecision == std::string::npos) continue;

            auto newName = filename.replace(havePrecision, precision.first.size(), "");
            newName = dirname + kPathSeparator + precision.second + kPathSeparator + newName;

            if (exist(newName)) {
                return newName;
            }
        }
    }

    // checking dirname
    auto getModelsDirname = [](std::string path) -> std::string {
        std::string dir = getDirname(path);

        struct stat sb;
        if (stat(dir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
            return "";
        }
        return dir;
    };

    for (auto & path : absModelsPath) {
        std::string publicDir = getModelsDirname(path);

        if (!publicDir.empty()) {
            return path;
        }
    }
    std::stringstream errorMsg;
    errorMsg<< "path to model invalid, models found at: \n";

    for (auto & path : absModelsPath) {
        errorMsg << path <<"\n";
    }
    errorMsg << "also searched by parent directory names: \n";
    for (auto & path : absModelsPath) {
        errorMsg << getDirname(path) << "\n";
    }

    std::cout << errorMsg.str();
    ::testing::AssertionFailure() << errorMsg.str();

    // doesn't matter what to return here
    return "";
}
