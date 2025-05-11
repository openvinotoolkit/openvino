// Copyright (C) 2018-2025 Intel Corporation
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

#include "openvino/openvino.hpp"

#include "samples/slog.hpp"
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
std::map<std::string, std::string> parseArgMap(std::string argMap);

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

void printInputAndOutputsInfo(const ov::Model& network);
ov::element::Type getPrecision2(const std::string& value);

template <class T>
void printInputAndOutputsInfoShort(const T& network) {
    slog::info << "Network inputs:" << slog::endl;
    for (auto&& input : network.inputs()) {
        std::string in_name;
        std::string node_name;

        // Workaround for "tensor has no name" issue
        try {
            for (const auto& name : input.get_names()) {
                in_name += name + " , ";
            }
            in_name = in_name.substr(0, in_name.size() - 3);
        } catch (const ov::Exception&) {
        }

        try {
            node_name = input.get_node()->get_friendly_name();
        } catch (const ov::Exception&) {
        }

        if (in_name == "") {
            in_name = "***NO_NAME***";
        }
        if (node_name == "") {
            node_name = "***NO_NAME***";
        }

        slog::info << "    " << in_name << " (node: " << node_name << ") : " << input.get_element_type() << " / "
                   << ov::layout::get_layout(input).to_string() << " / " << input.get_partial_shape() << slog::endl;
    }

    slog::info << "Network outputs:" << slog::endl;
    for (auto&& output : network.outputs()) {
        std::string out_name;
        std::string node_name;

        // Workaround for "tensor has no name" issue
        try {
            for (const auto& name : output.get_names()) {
                out_name += name + " , ";
            }
            out_name = out_name.substr(0, out_name.size() - 3);
        } catch (const ov::Exception&) {
        }
        try {
            node_name = output.get_node()->get_input_node_ptr(0)->get_friendly_name();
        } catch (const ov::Exception&) {
        }

        if (out_name == "") {
            out_name = "***NO_NAME***";
        }
        if (node_name == "") {
            node_name = "***NO_NAME***";
        }

        slog::info << "    " << out_name << " (node: " << node_name << ") : " << output.get_element_type() << " / "
                   << ov::layout::get_layout(output).to_string() << " / " << output.get_partial_shape() << slog::endl;
    }
}
