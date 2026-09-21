// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_utils/gpu_kernel_lifecycle.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace cldnn;

namespace {

class lifecycle_test_kernel final : public kernel {
public:
    explicit lifecycle_test_kernel(std::string id) : _id(std::move(id)) {}

    kernel::ptr clone(bool reuse_kernel_handle) const override {
        auto result = std::make_shared<lifecycle_test_kernel>(_id);
        result->_reused = reuse_kernel_handle;
        return result;
    }

    bool is_same(const kernel& other) const override {
        const auto* typed = dynamic_cast<const lifecycle_test_kernel*>(&other);
        return typed != nullptr && typed->_id == _id;
    }

    std::string get_id() const override {
        return _id;
    }
    std::vector<uint8_t> get_binary() const override {
        return {};
    }
    std::string get_build_log() const override {
        return {};
    }
    bool reused() const {
        return _reused;
    }

private:
    std::string _id;
    bool _reused = false;
};

}  // namespace

TEST(gpu_kernel_lifecycle, adopts_multiple_kernels_in_dispatch_order) {
    const auto first = std::make_shared<lifecycle_test_kernel>("first");
    const auto second = std::make_shared<lifecycle_test_kernel>("second");
    gpu_kernel_lifecycle lifecycle;

    lifecycle.adopt_entries({{second, 1}, {first, 0}});

    ASSERT_EQ(lifecycle.size(), 2u);
    EXPECT_EQ(lifecycle[0], first);
    EXPECT_EQ(lifecycle[1], second);
}

TEST(gpu_kernel_lifecycle, clones_kernel_handles_using_backend_reuse_policy) {
    const auto first = std::make_shared<lifecycle_test_kernel>("first");
    const auto second = std::make_shared<lifecycle_test_kernel>("second");
    gpu_kernel_lifecycle original;
    original.adopt_entries({{first, 0}, {second, 1}});

    gpu_kernel_lifecycle clone;
    clone.clone_from(original, true);

    ASSERT_EQ(clone.size(), 2u);
    EXPECT_NE(clone[0], original[0]);
    EXPECT_NE(clone[1], original[1]);
    EXPECT_TRUE(std::dynamic_pointer_cast<lifecycle_test_kernel>(clone[0])->reused());
    EXPECT_TRUE(std::dynamic_pointer_cast<lifecycle_test_kernel>(clone[1])->reused());
}
