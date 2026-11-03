// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_utils/prefetch_streambuf.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <istream>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "openvino/runtime/tensor.hpp"

TEST(PrefetchStreamBufTest, ReadsReadOnlyMappedRoiTensor) {
    constexpr size_t k_size = 64 * 1024;
    constexpr size_t k_header = 16;
    std::vector<char> content(k_size);
    for (size_t i = 0; i < k_size; ++i) {
        content[i] = static_cast<char>(i % 251u);
    }

    const auto path = std::filesystem::path(ov::test::utils::generateTestFilePrefix() + "_prefetch_streambuf.bin");
    {
        std::ofstream ofs(path, std::ios::binary);
        ASSERT_TRUE(ofs.write(content.data(), static_cast<std::streamsize>(k_size)));
    }
    {
        // Same shape as a cache blob: a read-only mmap tensor with the header cut off by a ROI.
        const auto blob = ov::read_tensor_data(path);
        const ov::Tensor roi{blob, {k_header}, {k_size}};

        ov::intel_gpu::PrefetchStreamBuf buf(roi);
        std::istream stream(&buf);
        std::vector<char> got(k_size - k_header);
        ASSERT_TRUE(stream.read(got.data(), static_cast<std::streamsize>(got.size())));
        EXPECT_EQ(got, std::vector<char>(content.begin() + k_header, content.end()));
    }
    std::filesystem::remove(path);
}
