// Copyright (C) 2018-2020 Intel Corporation
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
#include <vpu/utils/string.hpp>

#include <inference_engine.hpp>

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
void readInputFilesArguments(std::vector<std::string> &files, const std::string& arg);

/**
* @brief This function find -i/--images key in input args
*        It's necessary to process multiple values for single key
* @return files updated vector of verified input files
*/
void parseInputFilesArguments(std::vector<std::string> &files);

void processPrecisions(InferenceEngine::CNNNetwork& network, const std::string &ip, const std::string &op,
        const std::string &iop);

void printInputAndOutputsInfo(const InferenceEngine::CNNNetwork& network);

// TODO: can be removed from header as layout it put here as well
std::map<std::string, std::string> parseArgMap(std::string argMap);
