// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common samples functionality
 * @file args_helper.hpp
 */

#pragma once

// clang-format off
#include <string>
#include <vector>

#include "inference_engine.hpp"
#include "openvino/openvino.hpp"
// clang-format on

/**
 * @brief This function checks input args and existence of specified files in a given folder
 * @param arg path to a file to be checked for existence
 * @return files updated vector of verified input files
 */
void readInputFilesArguments(std::vector<std::string>& files, const std::string& arg);

/**
 * @brief This function find -i/--images key in input args
 *        It's necessary to process multiple values for single key
 * @return files updated vector of verified input files
 */
void parseInputFilesArguments(std::vector<std::string>& files);

void processPrecision(InferenceEngine::CNNNetwork& network,
                      const std::string& ip,
                      const std::string& op,
                      const std::string& iop);

void processLayout(InferenceEngine::CNNNetwork& network,
                   const std::string& il,
                   const std::string& ol,
                   const std::string& iol);

void printInputAndOutputsInfo(const InferenceEngine::CNNNetwork& network);
void printInputAndOutputsInfo(const ov::Model& network);

void configurePrePostProcessing(std::shared_ptr<ov::Model>& function,
                                const std::string& ip,
                                const std::string& op,
                                const std::string& iop,
                                const std::string& il,
                                const std::string& ol,
                                const std::string& iol,
                                const std::string& iml,
                                const std::string& oml,
                                const std::string& ioml);

//--- API 2.0 -------------------------------------------------------------------------
void printInputAndOutputsInfo(const ov::Model& network);
void printInputAndOutputsInfoShort(const ov::Model& network);
void processPrecision(const ov::Model& network, const std::string& ip, const std::string& op, const std::string& iop);
