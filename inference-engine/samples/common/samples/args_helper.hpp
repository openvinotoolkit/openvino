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

std::map<std::string, std::string> parseArgMap(std::string argMap) {
    argMap.erase(std::remove_if(argMap.begin(), argMap.end(), ::isspace), argMap.end());

    std::vector<std::string> pairs;
    vpu::splitStringList(argMap, pairs, ',');

    std::map<std::string, std::string> parsedMap;
    for (auto&& pair : pairs) {
        std::vector<std::string> keyValue;
        vpu::splitStringList(pair, keyValue, ':');
        if (keyValue.size() != 2) {
            throw std::invalid_argument("Invalid key/value pair " + pair + ". Expected <layer_name>:<value>");
        }

        parsedMap[keyValue[0]] = keyValue[1];
    }

    return parsedMap;
}

using supported_precisions_t = std::unordered_map<std::string, InferenceEngine::Precision>;

InferenceEngine::Precision getPrecision(std::string value,
                                               const supported_precisions_t& supported_precisions) {
    std::transform(value.begin(), value.end(), value.begin(), ::toupper);

    const auto precision = supported_precisions.find(value);
    if (precision == supported_precisions.end()) {
        throw std::logic_error("\"" + value + "\"" + " is not a valid precision");
    }

    return precision->second;
}

InferenceEngine::Precision getPrecision(const std::string& value) {
    static const supported_precisions_t supported_precisions = {
         { "FP32", InferenceEngine::Precision::FP32 },
         { "FP16", InferenceEngine::Precision::FP16 },
         { "BF16", InferenceEngine::Precision::BF16 },
         { "U64", InferenceEngine::Precision::U64 },
         { "I64", InferenceEngine::Precision::I64 },
         { "U32", InferenceEngine::Precision::U32 },
         { "I32", InferenceEngine::Precision::I32 },
         { "U16", InferenceEngine::Precision::U16 },
         { "I16", InferenceEngine::Precision::I16 },
         { "U8", InferenceEngine::Precision::U8 },
         { "I8", InferenceEngine::Precision::I8 },
         { "BOOL", InferenceEngine::Precision::BOOL },
    };

    return getPrecision(value, supported_precisions);
}

void setPrecisions(const InferenceEngine::CNNNetwork& network, const std::string &iop) {
    const auto user_precisions_map = parseArgMap(iop);

    auto inputs = network.getInputsInfo();
    auto outputs = network.getOutputsInfo();

    for (auto&& item : user_precisions_map) {
        const auto& layer_name = item.first;
        const auto& user_precision = item.second;

        const auto input = inputs.find(layer_name);
        const auto output = outputs.find(layer_name);

        if (input != inputs.end()) {
            input->second->setPrecision(getPrecision(user_precision));
        } else if (output != outputs.end()) {
            output->second->setPrecision(getPrecision(user_precision));
        } else {
            throw std::logic_error(layer_name + " is not an input neither output");
        }
    }
}

void processPrecisions(InferenceEngine::CNNNetwork& network, const std::string &ip, const std::string &op,
        const std::string &iop) {
    if (!ip.empty()) {
        const auto user_precision = getPrecision(ip);
        for (auto&& layer : network.getInputsInfo()) {
            layer.second->setPrecision(user_precision);
        }
    }

    if (!op.empty()) {
        auto user_precision = getPrecision(op);
        for (auto&& layer : network.getOutputsInfo()) {
            layer.second->setPrecision(user_precision);
        }
    }

    if (!iop.empty()) {
        setPrecisions(network, iop);
    }
}

void printInputAndOutputsInfo(const InferenceEngine::CNNNetwork& network) {
    std::cout << "Network inputs:" << std::endl;
    for (auto&& layer : network.getInputsInfo()) {
        std::cout << "    " << layer.first << " : " << layer.second->getPrecision() << " / " << layer.second->getLayout() << std::endl;
    }
    std::cout << "Network outputs:" << std::endl;
    for (auto&& layer : network.getOutputsInfo()) {
        std::cout << "    " << layer.first << " : " << layer.second->getPrecision() << " / " << layer.second->getLayout() << std::endl;
    }
}
