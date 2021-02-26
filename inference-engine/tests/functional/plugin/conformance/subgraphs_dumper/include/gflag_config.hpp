// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>
#include <iostream>

static const char help_message[] = "Print a usage message.";
static const char input_folders_message[] = "Required. Paths to the input folders with IRs";
static const char output_folders_message[] = "Required. Paths to the output folders with IRs";

DEFINE_bool(h, false, help_message);
DEFINE_string(input_folders, ".", input_folders_message);
DEFINE_string(output_folders, "output", output_folders_message);

/**
* @brief This function shows a help message
*/
static void showUsage() {
    std::cout << "\n";
    std::cout << "Subgraph Dumper [OPTION]\n";
    std::cout << "Options:\n";
    std::cout << "\n";
    std::cout << "    -h                         "  << help_message << "\n";
    std::cout << "    --input_folders \"<path>\" "  << input_folders_message << "\n";
    std::cout << "    --output_folders \"<path>\" " << output_folders_message << "\n";
}