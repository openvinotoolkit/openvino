// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gflags/gflags.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "metadata.hpp"
namespace {

static constexpr char help_message[] = "Optional. Print the usage message.";

static constexpr char blob_message[] = "Required. Path to the NPU generated blob (with metadata).";

static constexpr char output_message[] =
    "Optional. Path to the output file. Default value: \"raw_<npu_blob_file>.blob\".";

DEFINE_bool(h, false, help_message);
DEFINE_string(b, "", blob_message);
DEFINE_string(o, "", output_message);

static constexpr uint8_t MAX_PRINTABLE_VECTOR_ELEMENTS = 6;  // 1, 2, 3, ..., 7, 8, 9

template <typename T>
struct PrintHelper {
    PrintHelper(T& tmp) : _tmp(tmp) {}

    template <typename, typename = void>
    static constexpr bool has_to_string = false;

    friend std::ostream& operator<<(std::ostream& ostream, const PrintHelper<T>& other) {
        if constexpr (has_to_string<T>) {
            ostream << other._tmp.to_string();
        } else {
            ostream << other._tmp;
        }
        return ostream;
    }

    T& _tmp;
};

template <typename T>
template <typename U>
constexpr bool PrintHelper<T>::has_to_string<U, std::void_t<decltype(std::declval<U&>().to_string())>> = true;

template <typename T>
std::ostream& operator<<(std::ostream& ostream, const std::vector<T>& vec) {
    ostream << "(size = " << vec.size() << "): [";
    if (vec.size() <= MAX_PRINTABLE_VECTOR_ELEMENTS) {
        for (auto it = vec.begin(); it != vec.end() - 1; ++it) {
            ostream << PrintHelper(*it) << ", ";
        }
    } else {
        for (auto it = vec.begin(); it != vec.begin() + MAX_PRINTABLE_VECTOR_ELEMENTS / 2; ++it) {
            ostream << PrintHelper(*it) << ", ";
        }
        ostream << "..., ";
        for (auto it = vec.end() - MAX_PRINTABLE_VECTOR_ELEMENTS / 2; it != vec.end() - 1; ++it) {
            ostream << PrintHelper(*it) << ", ";
        }
    }
    ostream << PrintHelper(*(vec.end() - 1)) << "]";
    return ostream;
}

static void showUsage() {
    std::cout << "npu_metadata_parser [OPTIONS]" << std::endl;
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

std::pair<uint32_t, std::unique_ptr<intel_npu::MetadataBase>> parseNPUMetadata(std::istream& stream) {
    size_t magicBytesSize = intel_npu::MAGIC_BYTES.size();
    std::string blobMagicBytes;
    blobMagicBytes.resize(magicBytesSize);

    stream.read(blobMagicBytes.data(), magicBytesSize);
    if (intel_npu::MAGIC_BYTES != blobMagicBytes) {
        OPENVINO_THROW("Blob is missing NPU metadata!");
    }

    uint32_t metaVersion;
    stream.read(reinterpret_cast<char*>(&metaVersion), sizeof(metaVersion));

    std::unique_ptr<intel_npu::MetadataBase> storedMeta;
    storedMeta = intel_npu::create_metadata(metaVersion);
    storedMeta->read(stream);

    return {metaVersion, std::move(storedMeta)};
}

void printNPUMetadata(uint32_t version, const intel_npu::MetadataBase* metadataPtr) {
    std::cout << "NPU metadata version ........... " << intel_npu::MetadataBase::get_major(version) << "."
              << intel_npu::MetadataBase::get_minor(version) << std::endl;
    ;

    auto initSizesOpt = metadataPtr->get_init_sizes();
    if (initSizesOpt.has_value()) {
        std::cout << "Init sizes ........... " << initSizesOpt.value() << std::endl;
    }

    auto batchSizeOpt = metadataPtr->get_batch_size();
    if (batchSizeOpt.has_value()) {
        std::cout << "Batch size ........... " << batchSizeOpt.value() << std::endl;
    }

    auto inputLayoutsOpt = metadataPtr->get_input_layouts();
    if (inputLayoutsOpt.has_value()) {
        std::cout << "Input layouts ........... " << inputLayoutsOpt.value() << std::endl;
    }

    auto ouputLayoutsOpt = metadataPtr->get_input_layouts();
    if (ouputLayoutsOpt.has_value()) {
        std::cout << "Output layouts ........... " << ouputLayoutsOpt.value() << std::endl;
    }
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
        std::cout << "Parsing command-line arguments ........... ";
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
        auto [npuMetadataVersion, metadata] = parseNPUMetadata(blob);
        std::cout << "Done!" << std::endl;

        std::cout << "Printing metadata info from NPU blob ........... " << std::endl;
        printNPUMetadata(npuMetadataVersion, metadata.get());

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
