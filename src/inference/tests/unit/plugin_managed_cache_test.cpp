// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/any.hpp"
#include "openvino/runtime/iplugin_managed_cache.hpp"
#include "openvino/runtime/plugin_managed_cache.hpp"

namespace {

using ::testing::HasSubstr;

/// One KEY_VALUE group over a block counter; records the calls.
class FakeCache : public ov::IPluginManagedCache {
public:
    std::vector<ov::CacheGroupInfo> get_groups() const override {
        ov::CacheGroupInfo group;
        group.kind = ov::CacheKind::KEY_VALUE;
        group.tensors = {{"key_cache.0", 0, ov::element::u8, ov::PartialShape{-1, 8, 16, 64}},
                         {"value_cache.0", 0, ov::element::u8, ov::PartialShape{-1, 8, 16, 64}}};
        group.block_size = 16;
        group.block_size_in_bytes = 2 * 8 * 16 * 64;
        return {group};
    }
    size_t get_available_memory() const override {
        return 1u << 20;
    }
    void allocate(size_t group, size_t num_blocks, bool zero) override {
        check(group);
        OPENVINO_ASSERT(blocks == 0, "already allocated");
        blocks = num_blocks;
        zeroed = zero;
    }
    void reallocate(size_t group, size_t num_blocks, bool zero) override {
        check(group);
        blocks = num_blocks;
        zeroed = zero;
    }
    void deallocate(size_t group) override {
        check(group);
        blocks = 0;
    }
    size_t get_num_allocated_blocks(size_t group) const override {
        check(group);
        return blocks;
    }
    void copy_blocks(size_t group, const std::map<size_t, std::list<size_t>>& map) override {
        check(group);
        copies = map;
    }
    void zero_blocks(size_t group, const std::set<size_t>& indices) override {
        check(group);
        zeroed_blocks = indices;
    }

    size_t blocks = 0;
    bool zeroed = false;
    std::map<size_t, std::list<size_t>> copies;
    std::set<size_t> zeroed_blocks;

private:
    static void check(size_t group) {
        OPENVINO_ASSERT(group == 0, "no cache group ", group);
    }
};

TEST(PluginManagedCacheTest, ForwardsEveryCallToTheImplementation) {
    auto impl = std::make_shared<FakeCache>();
    ov::PluginManagedCache cache(impl);
    ASSERT_TRUE(static_cast<bool>(cache));

    const auto groups = cache.get_groups();
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].kind, ov::CacheKind::KEY_VALUE);
    EXPECT_EQ(groups[0].tensors.at(1).name, "value_cache.0");
    EXPECT_EQ(cache.get_available_memory(), 1u << 20);

    cache.allocate(0, 4, true);
    EXPECT_EQ(cache.get_num_allocated_blocks(0), 4u);
    EXPECT_TRUE(impl->zeroed);
    cache.reallocate(0, 9, false);
    EXPECT_EQ(cache.get_num_allocated_blocks(0), 9u);
    cache.copy_blocks(0, {{1, {2, 3}}});
    EXPECT_EQ(impl->copies, (std::map<size_t, std::list<size_t>>{{1, {2, 3}}}));
    cache.zero_blocks(0, {5, 6});
    EXPECT_EQ(impl->zeroed_blocks, (std::set<size_t>{5, 6}));
    cache.deallocate(0);
    EXPECT_EQ(cache.get_num_allocated_blocks(0), 0u);
}

TEST(PluginManagedCacheTest, ReportsImplementationErrorsAsOvExceptions) {
    ov::PluginManagedCache cache(std::make_shared<FakeCache>());
    cache.allocate(0, 4, false);
    OV_EXPECT_THROW(cache.allocate(0, 4, false), ov::Exception, HasSubstr("already allocated"));
    OV_EXPECT_THROW(cache.get_num_allocated_blocks(1), ov::Exception, HasSubstr("no cache group 1"));
}

TEST(PluginManagedCacheTest, AnEmptyFacadeRefusesEveryCall) {
    ov::PluginManagedCache cache;
    EXPECT_FALSE(static_cast<bool>(cache));
    OV_EXPECT_THROW(cache.get_groups(), ov::Exception, HasSubstr("PluginManagedCache was not initialized"));
    OV_EXPECT_THROW(cache.deallocate(0), ov::Exception, HasSubstr("PluginManagedCache was not initialized"));
    OV_EXPECT_THROW(ov::PluginManagedCache(nullptr), ov::Exception, HasSubstr("was not initialized"));
}

TEST(PluginManagedCacheTest, TravelsAsAPropertyValue) {
    auto impl = std::make_shared<FakeCache>();
    ov::Any value = ov::PluginManagedCache(impl);
    auto cache = value.as<ov::PluginManagedCache>();
    cache.allocate(0, 3, false);
    EXPECT_EQ(impl->blocks, 3u);
    EXPECT_EQ(std::string(ov::plugin_managed_cache.name()), "PLUGIN_MANAGED_CACHE");
}

}  // namespace
