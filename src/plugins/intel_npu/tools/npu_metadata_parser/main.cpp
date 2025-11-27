// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gflags/gflags.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "metadata_helper.hpp"

namespace {

static constexpr char help_message[] = "Optional. Print the usage message.";

static constexpr char blob_message[] = "Required. Path to the NPU generated blob (with metadata).";

static constexpr char output_message[] =
    "Optional. Path to the output file. Default value: \"raw_<npu_blob_file>.blob\".";

DEFINE_bool(h, false, help_message);
DEFINE_string(b, "", blob_message);
DEFINE_string(o, "", output_message);

static void showUsage() {
    std::cout << "raw_blob_extractor [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << " Common options:                             " << std::endl;
    std::cout << "    -h                                       " << help_message << std::endl;
    std::cout << "    -b                           <value>     " << blob_message << std::endl;
    std::cout << "    -o                           <value>     " << output_message << std::endl;
    std::cout << std::endl;
}

static bool parseCommandLine(int* argc, char*** argv) {
    gflags::ParseCommandLineNonHelpFlags(argc, argv, true);

    if (FLAGS_h) {
        showUsage();
        return false;
    }

    if (FLAGS_b.empty()) {
        throw std::invalid_argument("Path to NPU generated compiled model blob is required");
    }

    if (1 < *argc) {
        std::stringstream message;
        message << "Unknown arguments: ";
        for (auto arg = 1; arg < *argc; arg++) {
            message << (*argv)[arg];
            if (arg < *argc) {
                message << " ";
            }
        }
        throw std::invalid_argument(message.str());
    }

    return true;
}

}  // namespace

using TimeDiff = std::chrono::milliseconds;

int main(int argc, char* argv[]) {
    try {
        // Steps in extracting NPU raw blob
        // 1. Parse command line arguments
        // 2. Read metatada at the beginning of the blob
        // 3. Print metadata (version, capabilities etc.)
        // 4. Read raw blob at the end of metadata
        // 5. Export raw blob to file
        TimeDiff parseMetadataWriteBlobTimeElapsed{0};

        const auto& version = ov::get_openvino_version();
        std::cout << version.description << " version ......... ";
        std::cout << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "." << OPENVINO_VERSION_PATCH
                  << std::endl;

        std::cout << "Build ........... ";
        std::cout << version.buildNumber << std::endl;
        std::cout << "Parsing command-line arguments ........... " << std::endl;
        if (!parseCommandLine(&argc, &argv)) {
            return EXIT_SUCCESS;
        }
        std::cout << "Done!" << std::endl;

        std::cout << "Loading NPU blob ........... ";
        std::ifstream blob(FLAGS_b, std::ios::in | std::ios::binary);
        std::ofstream rawBlob(FLAGS_o, std::ios::out | std::ios::binary);
        if (!blob) {
            throw std::runtime_error("Could not read from blob file!");
        } else if (!rawBlob) {
            throw std::runtime_error("Could not write to output raw blob file!");
        }
        std::cout << "Done!" << std::endl;

        std::cout << "Parsing metadata ........... ";
        auto timeBeforeParsingMetadata = std::chrono::steady_clock::now();
        auto [npuMetadataVersion, metadata] = npu::utils::parseNPUMetadata(blob);
        std::cout << "Done!" << std::endl;

        std::cout << "Printing metadata info from NPU blob ........... " << std::endl;
        npu::utils::printNPUMetadata(npuMetadataVersion, metadata.get());

        std::cout << "Writing raw blob to output file ........... ";
        rawBlob << blob.rdbuf();
        std::cout << "Done!" << std::endl;

        parseMetadataWriteBlobTimeElapsed =
            std::chrono::duration_cast<TimeDiff>(std::chrono::steady_clock::now() - timeBeforeParsingMetadata);
        std::cout << "Done. Time elapsed: " << parseMetadataWriteBlobTimeElapsed.count() << " ms" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
