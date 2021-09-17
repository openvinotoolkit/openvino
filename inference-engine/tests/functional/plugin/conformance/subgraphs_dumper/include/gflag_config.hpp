// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>
#include <iostream>

static const char help_message[] = "Print a usage message.";
static const char input_folders_message[] = "Required. Comma separated paths to the input folders with IRs";
static const char local_cache_message[] = "Optional. Comma separated paths to the local cache folders with IRs";
static const char output_folder_message[] = "Required. Path to the output folders where to serialize IRs";
static const char path_regex_message[] = "Optional. regular expression to be applied in input "
                                         "folders recursive discovery";
static const char constants_size_threshold_message[] = "Optional. Maximum size of constant in megabytes"
                                                       " to be serialized.\n"
                                                       "If constant size exceeds specified number it will be replaced"
                                                       "with parameter and meta information about original data range "
                                                       "will be saved";
static const char eliminate_dynamism_message[] = "Optional. If specified dynamic shapes will be eliminated from model"
                                                 "and replaced by propagated upper bound values (if possible)";

DEFINE_bool(h, false, help_message);
DEFINE_string(input_folders, "", local_cache_message);
DEFINE_string(local_cache, ".", input_folders_message);
DEFINE_string(output_folder, "output", output_folder_message);
DEFINE_string(path_regex, ".*", output_folder_message);
DEFINE_double(constants_size_threshold, 1., constants_size_threshold_message);
DEFINE_bool(eliminate_dynamism, true, eliminate_dynamism_message);

/**
* @brief This function shows a help message
*/
static void showUsage() {
    std::cout << "\n";
    std::cout << "Subgraph Dumper [OPTION]\n";
    std::cout << "Options:\n";
    std::cout << "\n";
    std::cout << "    -h                                     " << help_message << "\n";
    std::cout << "    --input_folders \"<path>\"             " << input_folders_message << "\n";
    std::cout << "    --local_cache \"<path>\"               " << input_folders_message << "\n";
    std::cout << "    --output_folder \"<path>\"             " << output_folder_message << "\n";
    std::cout << "    --path_regex \"<path>\"                " << path_regex_message << "\n";
    std::cout << "    --constants_size_threshold \"<value>\" " << constants_size_threshold_message << "\n";
    std::cout << "    --eliminate_dynamism \"<value>\"       " << eliminate_dynamism_message << "\n";
    std::cout << std::flush;
}