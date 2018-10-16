// Copyright (C) 2018 Intel Corporation
//
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

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/**
* @brief This function check input args and find images in given folder
*/
void readImagesArguments(std::vector<std::string> &images, const std::string& arg) {
    struct stat sb;
    if (stat(arg.c_str(), &sb) != 0) {
        std::cout << "[ WARNING ] File " << arg << " cannot be opened!" << std::endl;
        return;
    }
    if (S_ISDIR(sb.st_mode)) {
        DIR *dp;
        dp = opendir(arg.c_str());
        if (dp == nullptr) {
            std::cout << "[ WARNING ] Directory " << arg << " cannot be opened!" << std::endl;
            return;
        }

        struct dirent *ep;
        while (nullptr != (ep = readdir(dp))) {
            std::string fileName = ep->d_name;
            if (fileName == "." || fileName == "..") continue;
            std::cout << "[ INFO ] Add file  " << ep->d_name << " from directory " << arg << "." << std::endl;
            images.push_back(arg + "/" + ep->d_name);
        }
    } else {
        images.push_back(arg);
    }
}

/**
* @brief This function find -i/--images key in input args
*        It's necessary to process multiple values for single key
*/
void parseImagesArguments(std::vector<std::string> &images) {
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
        readImagesArguments(images, args.at(i));
    }
}
