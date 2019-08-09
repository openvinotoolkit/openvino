// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common samples functionality
 * @file args_helper.hpp
 */

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>
#include <sys/stat.h>

#include <samples/slog.hpp>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/**
* @brief This function checks input args and existence of specified files in a given folder
* @param arg path to a file to be checked for existence
* @return files updated vector of verified input files
*/
void readInputFilesArguments(std::vector<std::string> &files, const std::string& arg) {
    struct stat sb;
    if (stat(arg.c_str(), &sb) != 0) {
        slog::warn << "File " << arg << " cannot be opened!" << slog::endl;
        return;
    }
    if (S_ISDIR(sb.st_mode)) {
        DIR *dp;
        dp = opendir(arg.c_str());
        if (dp == nullptr) {
            slog::warn << "Directory " << arg << " cannot be opened!" << slog::endl;
            return;
        }

        struct dirent *ep;
        while (nullptr != (ep = readdir(dp))) {
            std::string fileName = ep->d_name;
            if (fileName == "." || fileName == "..") continue;
            files.push_back(arg + "/" + ep->d_name);
        }
        closedir(dp);
    } else {
        files.push_back(arg);
    }

    if (files.size() < 20) {
        slog::info << "Files were added: " << files.size() << slog::endl;
        for (std::string filePath : files) {
            slog::info << "    " << filePath << slog::endl;
        }
    } else {
        slog::info << "Files were added: " << files.size() << ". Too many to display each of them." << slog::endl;
    }
}

/**
* @brief This function find -i/--images key in input args
*        It's necessary to process multiple values for single key
* @return files updated vector of verified input files
*/
void parseInputFilesArguments(std::vector<std::string> &files) {
    std::vector<std::string> args = gflags::GetArgvs();
    bool readArguments = false;
    for (size_t i = 0; i < args.size(); i++) {
        if (args.at(i) == "-i" || args.at(i) == "--images") {
            readArguments = true;
            continue;
        }
        if (!readArguments) {
            continue;
        }
        if (args.at(i).c_str()[0] == '-') {
            break;
        }
        readInputFilesArguments(files, args.at(i));
    }
}
