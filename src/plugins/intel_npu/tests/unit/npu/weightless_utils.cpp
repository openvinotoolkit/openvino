// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weightless_utils.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "intel_npu/common/network_metadata.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace intel_npu;

TEST(WeightlessUtilsOverflow, CraftedShapeSizeOverflowIsRejected) {
    // Write a tiny weights file so mapped_memory->size() is small.
    const std::string weightsPath = "crafted_overflow_weights.bin";
    struct TempFileCleanup final {
        explicit TempFileCleanup(const char* file_path) : path{file_path} {}
        ~TempFileCleanup() {
            std::remove(path);
        }
        const char* path;
    } cleanup{weightsPath.c_str()};

    {
        std::ofstream ofs(weightsPath, std::ios::binary);
        const std::vector<char> bytes(16, 0);  // 16-byte mmap-able file
        ofs.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    IODescriptor descriptor;
    // id must parse from nameFromCompiler (view_to_number<size_t>) and be within file.
    descriptor.nameFromCompiler = "0";
    descriptor.precision = ov::element::f32;
    // This shape is chosen to overflow unchecked byte-size calculations.
    descriptor.shapeFromCompiler = ov::PartialShape({1ll << 32, 1ll << 32, 4});
    descriptor.isInitInputWeights = true;
    descriptor.indexUsedByDriver = 0;

    NetworkMetadata meta;
    meta.name = "init_schedule";
    meta.inputs.push_back(descriptor);

    std::vector<NetworkMetadata> initNetworkMetadata{meta};

    // Overflow should be detected and rejected.
    EXPECT_THROW(get_all_constants_memory_mapped(weightsPath, initNetworkMetadata), ov::Exception);
}
