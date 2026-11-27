// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/xml_util/weights_provider.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

#include "openvino/runtime/aligned_buffer.hpp"

namespace ov::test {

TEST(WeightsProviderTest, buffer_provider_without_origin_has_empty_path) {
    const ov::util::BufferWeightsProvider provider(std::make_shared<ov::AlignedBuffer>(16));

    EXPECT_TRUE(provider.path().empty());
    EXPECT_EQ(provider.size(), 16u);
}

TEST(WeightsProviderTest, buffer_provider_keeps_origin_path) {
    const std::filesystem::path origin{"model.bin"};
    const ov::util::BufferWeightsProvider provider(std::make_shared<ov::AlignedBuffer>(16), origin);

    EXPECT_EQ(provider.path(), origin);
    EXPECT_EQ(provider.size(), 16u);
}

TEST(WeightsProviderTest, buffer_provider_origin_path_is_visible_through_the_interface) {
    const std::filesystem::path origin{"model.bin"};
    const std::shared_ptr<ov::util::WeightsProvider> provider =
        std::make_shared<ov::util::BufferWeightsProvider>(std::make_shared<ov::AlignedBuffer>(16), origin);

    EXPECT_EQ(provider->path(), origin);
}

}  // namespace ov::test
