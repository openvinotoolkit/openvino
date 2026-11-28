// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "intel_npu/common/icontext.hpp"

using intel_npu::IContext;

namespace {

// What a test writes to get a context without a driver: a dummy handle and a way to drop it
class FakeContext final : public IContext {
public:
    void* getContextHandle() const override {
        return _context;
    }

    // Simulates the context being destroyed, for example after the device was lost
    void destroyContext() {
        _context = nullptr;
    }

private:
    void* _context = reinterpret_cast<void*>(0x3);
};

}  // namespace

TEST(Context, IsAliveWhileTheContextHandleIsSet) {
    FakeContext context;

    EXPECT_NE(context.getContextHandle(), nullptr);
    EXPECT_TRUE(context.isAlive());
}

TEST(Context, IsNotAliveOnceTheContextIsDestroyed) {
    FakeContext context;

    context.destroyContext();

    EXPECT_EQ(context.getContextHandle(), nullptr);
    EXPECT_FALSE(context.isAlive());
}

TEST(Context, HandleIsAvailableThroughTheInterface) {
    const std::shared_ptr<IContext> context = std::make_shared<FakeContext>();

    EXPECT_EQ(context->getContextHandle(), reinterpret_cast<void*>(0x3));
}

TEST(Context, MemoryIsNotReportedUnlessTheContextOverridesIt) {
    FakeContext context;

    EXPECT_ANY_THROW(context.getAllocMemSize());
    EXPECT_ANY_THROW(context.getTotalMemSize());
}

namespace {

// A context that can report its memory
class MemoryContext final : public IContext {
public:
    void* getContextHandle() const override {
        return reinterpret_cast<void*>(0x2);
    }
    uint64_t getAllocMemSize() const override {
        return 1024;
    }
    uint64_t getTotalMemSize() const override {
        return 4096;
    }
};

}  // namespace

TEST(Context, MemoryIsReportedThroughTheInterface) {
    const std::shared_ptr<IContext> context = std::make_shared<MemoryContext>();

    EXPECT_EQ(context->getAllocMemSize(), 1024u);
    EXPECT_EQ(context->getTotalMemSize(), 4096u);
    EXPECT_LT(context->getAllocMemSize(), context->getTotalMemSize());
}
