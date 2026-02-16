// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/weight_sharing_util.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/variant_visitor.hpp"

namespace ov::test {

using ov::op::v0::Parameter, ov::op::v0::Constant, ov::op::v1::Add;

class WeightShareExtensionTest : public testing::Test {
protected:
    void SetUp() override {
        ov::util::create_directory_recursive(test_dir);
    }

    void TearDown() override {
        if (util::directory_exists(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
    }

    void store_weights(const std::filesystem::path& path, const void* data, size_t size) {
        if (std::ofstream out_file(path, std::ios::binary); out_file.is_open()) {
            out_file.write(reinterpret_cast<const char*>(data), size);
        } else {
            FAIL() << "Failed to open file for writing: " << path;
        }
    }

    void create_test_weights_file(const std::filesystem::path& path, size_t size = 4000) {
        auto weights = ov::Tensor(ov::element::f32, Shape{size});
        std::generate_n(weights.data<float>(), weights.get_size(), [n = 0.0f]() mutable {
            return n++;
        });
        store_weights(path, weights.data(), weights.get_byte_size());
    }

    std::filesystem::path test_dir = utils::generateTestFilePrefix();
};

// test descriptor
class MappedMemDescriptor final : public ov::IBufferDescriptor {
public:
    explicit MappedMemDescriptor(std::shared_ptr<ov::MappedMemory> mem, size_t offset = 0)
        : m_id{mem->get_id()},
          m_offset(offset),
          m_mem{mem} {}

    uint64_t get_id() const override {
        return m_id;
    }

    size_t get_offset() const override {
        return m_offset;
    }

    std::shared_ptr<ov::AlignedBuffer> get_source_buffer() const override {
        if (auto mem = m_mem.lock()) {
            return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
                mem->data(),
                mem->size(),
                mem,
                std::make_shared<MappedMemDescriptor>(mem, m_offset));
        } else {
            return nullptr;
        }
    }

private:
    size_t m_id{0};
    size_t m_offset{0};
    std::weak_ptr<ov::MappedMemory> m_mem;
};

class BuffDesc final : public ov::IBufferDescriptor {
public:
    explicit BuffDesc(std::shared_ptr<ov::AlignedBuffer> mem, size_t offset)
        : m_id{mem->get_descriptor()->get_id()},
          m_offset(offset),
          m_mem{mem} {}

    explicit BuffDesc(std::shared_ptr<ov::AlignedBuffer> mem, size_t id, size_t offset)
        : m_id{id},
          m_offset(offset),
          m_mem{mem} {}

    uint64_t get_id() const override {
        return m_id;
    }

    size_t get_offset() const override {
        return m_offset;
    }

    std::shared_ptr<ov::AlignedBuffer> get_source_buffer() const override {
        if (auto mem = m_mem.lock(); mem && mem->get_descriptor()) {
            return mem->get_descriptor()->get_source_buffer();
        } else {
            return nullptr;
        }
    }

private:
    size_t m_id{0};
    size_t m_offset{0};
    std::weak_ptr<ov::AlignedBuffer> m_mem;
};

const auto create_test_model = []() {
    auto weights = ov::Tensor(ov::element::f32, Shape{2, 200});
    std::generate_n(weights.data<float>(), weights.get_size(), [n = 0.0f]() mutable {
        return n++;
    });

    auto param = std::make_shared<Parameter>(ov::element::f32, weights.get_shape());
    auto add = std::make_shared<Add>(param, std::make_shared<Constant>(weights));
    return std::make_shared<ov::Model>(add->outputs(), "TestModel");
};

const auto create_test_model_weights_from_file = [](const std::filesystem::path& weights_path) {
    auto w_buff = load_mmap_object(weights_path);
    auto w1 =
        std::make_shared<SharedBuffer<std::shared_ptr<MappedMemory>>>(w_buff->data(),
                                                                      w_buff->size() / 2,
                                                                      w_buff,
                                                                      std::make_shared<MappedMemDescriptor>(w_buff));
    auto w2 = std::make_shared<SharedBuffer<std::shared_ptr<MappedMemory>>>(
        w_buff->data() + w_buff->size() / 2,
        w_buff->size() / 2,
        w_buff,
        std::make_shared<MappedMemDescriptor>(w_buff, w_buff->size() / 2));
    auto param = std::make_shared<Parameter>(ov::element::f32, Shape{2, 200});
    auto add = std::make_shared<Add>(param, std::make_shared<Constant>(element::f32, Shape{1, 200}, w1));
    add = std::make_shared<Add>(add, std::make_shared<Constant>(element::f32, Shape{1, 200}, w2));
    return std::make_shared<ov::Model>(add->outputs(), "Test model weight from file");
};

TEST_F(WeightShareExtensionTest, get_constant_id_no_descriptor) {
    auto constant = Constant(element::i64, Shape{2, 2}, std::vector<int64_t>{1, 2, 3, 4});

    EXPECT_EQ(weight_sharing::Extension::get_constant_source_id(constant), weight_sharing::INVALID_SOURCE_ID);
    EXPECT_EQ(weight_sharing::Extension::get_constant_id(constant), weight_sharing::INVALID_CONSTANT_ID);
}

TEST_F(WeightShareExtensionTest, get_constant_id_with_descriptor) {
    const auto w_path = test_dir / "weights.bin";
    const auto id = std::hash<std::decay_t<decltype(w_path.native())>>{}(w_path.native());
    create_test_weights_file(w_path);
    auto w_buffer = load_mmap_object(w_path);

    auto w1 = std::make_shared<SharedBuffer<std::shared_ptr<MappedMemory>>>(
        w_buffer->data() + 200,
        w_buffer->size() - 200,
        w_buffer,
        std::make_shared<MappedMemDescriptor>(w_buffer, 200));
    auto constant = Constant(element::f32, Shape{200}, w1);

    EXPECT_EQ(weight_sharing::Extension::get_constant_source_id(constant), id);
    EXPECT_EQ(weight_sharing::Extension::get_constant_id(constant), 200);
}

TEST_F(WeightShareExtensionTest, get_constant_sources_for_model_with_no_source_tags) {
    auto model = create_test_model();
    auto weight_sources = weight_sharing::Extension::make_weight_sources(*model);
    EXPECT_TRUE(weight_sources.empty());
}

TEST_F(WeightShareExtensionTest, get_constant_sources_for_model_with_source_tags) {
    const auto weights_path = test_dir / "weights.bin";
    create_test_weights_file(weights_path);

    auto model = create_test_model_weights_from_file(weights_path);
    auto weight_sources = weight_sharing::Extension::make_weight_sources(*model);
    EXPECT_GE(weight_sources.size(), 1);
}

TEST_F(WeightShareExtensionTest, make_constant_map_for_model_with_no_source_tags) {
    auto model = create_test_model();
    auto constant_map = weight_sharing::Extension::make_constant_map(*model);
    EXPECT_TRUE(constant_map.empty());
}

TEST_F(WeightShareExtensionTest, make_constant_map_model_has_tagged_data_source) {
    const auto weights_path = test_dir / "weights.bin";
    create_test_weights_file(weights_path);

    auto model = create_test_model_weights_from_file(weights_path);
    auto constant_map = weight_sharing::Extension::make_constant_map(*model);
    EXPECT_GE(constant_map.size(), 1);
}

TEST_F(WeightShareExtensionTest, get_constant_origin_desc_no_wl_set) {
    auto c = Constant(element::f32, Shape{2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    ASSERT_FALSE(weight_sharing::Extension::get_constant_origin(c).has_value());
}

TEST_F(WeightShareExtensionTest, set_mapped_weight_buffer) {
    const auto weights_path = test_dir / "weights.bin";
    create_test_weights_file(weights_path);
    weight_sharing::Context shared_ctx;

    auto buffer = load_mmap_object(weights_path);
    ASSERT_TRUE(weight_sharing::set_weight_source(shared_ctx, buffer));
    EXPECT_EQ(shared_ctx.m_weight_sources.size(), 1);
}

TEST_F(WeightShareExtensionTest, set_aligned_weight_buffer) {
    weight_sharing::Context shared_ctx;
    auto buffer = std::make_shared<ov::AlignedBuffer>(4000);
    auto wt_buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        buffer->get_ptr<char>(),
        buffer->size(),
        buffer,
        std::make_shared<BuffDesc>(buffer, 12, 0));

    ASSERT_TRUE(weight_sharing::set_weight_source(shared_ctx, wt_buffer));
    EXPECT_EQ(shared_ctx.m_weight_sources.size(), 1);
    EXPECT_EQ(shared_ctx.m_weight_sources.count(12), 1);
}

TEST_F(WeightShareExtensionTest, set_null_aligned_weight_buffer) {
    weight_sharing::Context shared_ctx;
    std::shared_ptr<ov::AlignedBuffer> buffer = nullptr;
    ASSERT_FALSE(weight_sharing::set_weight_source(shared_ctx, buffer));
}

TEST_F(WeightShareExtensionTest, set_aligned_weight_buffer_no_tag) {
    weight_sharing::Context shared_ctx;
    auto buffer = std::make_shared<ov::AlignedBuffer>(4000);

    ASSERT_FALSE(weight_sharing::set_weight_source(shared_ctx, buffer));
}

TEST_F(WeightShareExtensionTest, set_aligned_weight_buffer_invalid_tag) {
    weight_sharing::Context shared_ctx;
    auto buffer = std::make_shared<ov::AlignedBuffer>(4000);

    ASSERT_FALSE(weight_sharing::set_weight_source(shared_ctx, buffer));
}

TEST_F(WeightShareExtensionTest, set_constant_buffer_with_no_id) {
    weight_sharing::Context shared_ctx;

    auto c = Constant(element::f32, Shape{2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    auto buffer = std::make_shared<ov::AlignedBuffer>(4000);

    ASSERT_FALSE(weight_sharing::set_constant(shared_ctx, c));
}

TEST_F(WeightShareExtensionTest, set_constant_buffer_with_id) {
    weight_sharing::Context shared_ctx;

    auto buffer = std::make_shared<ov::AlignedBuffer>(4000);
    auto wt_buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        buffer->get_ptr<char>() + 100,
        buffer->size(),
        buffer,
        std::make_shared<BuffDesc>(buffer, 12, 100));
    auto c = Constant(element::f32, Shape{100}, wt_buffer);

    ASSERT_TRUE(weight_sharing::set_constant(shared_ctx, c));
    const auto& [const_offset, const_size, const_type] = shared_ctx.m_constants_meta_data[12][100];
    EXPECT_EQ(const_offset, 100);
    EXPECT_EQ(const_size, 4000);
    EXPECT_EQ(const_type, element::f32);
}

TEST_F(WeightShareExtensionTest, get_origin_meta_data_from_constant) {
    auto c = Constant(element::f32, Shape{2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    c.get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute{32, 0, element::f64};

    const auto origin_desc = weight_sharing::Extension::get_constant_origin(c);

    ASSERT_TRUE(origin_desc.has_value());
    EXPECT_EQ(origin_desc->m_id, weight_sharing::INVALID_SOURCE_ID);
    EXPECT_EQ(origin_desc->m_offset, 0);
    EXPECT_EQ(origin_desc->m_size, 32);
    EXPECT_EQ(origin_desc->m_type, element::f64);
}

TEST_F(WeightShareExtensionTest, get_constant_buffer_for_null_weights) {
    weight_sharing::Context shared_ctx;

    const auto weights_path = test_dir / "weights.bin";
    create_test_weights_file(weights_path);
    auto wt_buffer = load_mmap_object(weights_path);
    weight_sharing::set_weight_source(shared_ctx, wt_buffer);
    auto c = Constant(element::f32,
                      Shape{10},
                      std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
                          wt_buffer->data(),
                          4 * 10,
                          wt_buffer,
                          std::make_shared<MappedMemDescriptor>(wt_buffer)));
    weight_sharing::set_constant(shared_ctx, c);

    auto constant_buffer = weight_sharing::get_buffer(shared_ctx, wt_buffer, 0);
    ASSERT_TRUE(constant_buffer);

    std::shared_ptr<ov::MappedMemory> wt_null;
    constant_buffer = weight_sharing::get_buffer(shared_ctx, wt_null, 0);
    ASSERT_FALSE(constant_buffer);
}

TEST_F(WeightShareExtensionTest, rebuild_constant_from_shared_context) {
    const auto weights_path = test_dir / "weights.bin";
    create_test_weights_file(weights_path);

    // create shared context with constant map for model
    uint64_t source_id_a{};
    const weight_sharing::ConstantMetaData const_a_1{0, 200, element::f32};
    const weight_sharing::ConstantMetaData const_a_2{200, 20, element::f32};

    weight_sharing::Context shared_ctx;

    // create model, which has been stored to blob
    auto [model, ref_data] = [&]() -> std::tuple<std::shared_ptr<ov::Model>, std::vector<float>> {
        auto w_mmap = load_mmap_object(weights_path);
        source_id_a = w_mmap->get_id();
        auto w_buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<MappedMemory>>>(
            w_mmap->data(),
            w_mmap->size(),
            w_mmap,
            std::make_shared<MappedMemDescriptor>(w_mmap));
        // plugin can created shared context for model and store buffer with tag
        weight_sharing::set_weight_source(shared_ctx, w_mmap);

        const auto& [c1_offset, c1_size, c1_type] = const_a_1;
        auto w1 = std::make_shared<SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
            w_buffer->get_ptr<char>() + c1_offset,
            c1_size,
            w_buffer,
            std::make_shared<BuffDesc>(w_buffer, c1_offset));

        const auto& [c2_offset, c2_size, c2_type] = const_a_2;
        auto w2 = std::make_shared<SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
            w_buffer->get_ptr<char>() + c2_offset,
            c2_size,
            w_buffer,
            std::make_shared<BuffDesc>(w_buffer, c2_offset));

        auto c1 = std::make_shared<Constant>(c1_type, Shape{5, 10}, w1);
        auto c2 = std::make_shared<Constant>(c2_type, Shape{5, 1}, w2);
        weight_sharing::set_constant(shared_ctx, *c1);
        weight_sharing::set_constant(shared_ctx, *c2);

        auto param = std::make_shared<Parameter>(ov::element::f32, Shape{1, 1});
        auto add = std::make_shared<Add>(param, c2);

        return std::make_tuple(std::make_shared<ov::Model>(add->outputs(), "Test model weight from file"),
                               c2->cast_vector<float>());
    }();

    // plugin build constant from shared context, weight buffer exists in memory
    {
        // the constant meta data read from blob (source ID, constant ID, shape)
        const auto const_src_id_from_blob = source_id_a;
        const auto const_id_from_blob = 200;
        const auto shape_from_blob = Shape{5, 1};

        // re-create constant buffer from shared context by IDs
        auto constant_buffer = weight_sharing::get_buffer(shared_ctx, const_src_id_from_blob, const_id_from_blob);
        ASSERT_TRUE(constant_buffer);
        const auto& c_type = shared_ctx.m_constants_meta_data[const_src_id_from_blob][const_id_from_blob].m_type;
        auto c = std::make_shared<Constant>(c_type, shape_from_blob, constant_buffer);

        EXPECT_EQ(c->cast_vector<float>(), ref_data);
        EXPECT_EQ(weight_sharing::Extension::get_constant_source_id(*c), const_src_id_from_blob);
        EXPECT_EQ(weight_sharing::Extension::get_constant_id(*c), const_id_from_blob);
    }
    // release model the shared context buffer will be not available
    model.reset();
    // plugin build constant from shared context but weight buffer has been released
    {
        // the constant meta data read from blob (source ID, constant ID, shape)
        const auto const_src_id_from_blob = source_id_a;
        const auto const_id_from_blob = 200;
        const auto shape_from_blob = Shape{5, 1};

        // get constant source buffer fail as not exists in memory
        auto weight_buffer = ov::wsh::get_source_buffer(shared_ctx, const_src_id_from_blob);
        ASSERT_FALSE(weight_buffer);
        auto constant_buffer = ov::wsh::get_buffer(shared_ctx, const_src_id_from_blob, const_id_from_blob);
        ASSERT_FALSE(constant_buffer);

        // constant can not be rebuild from shared context by IDs, restore manually
        const auto& c_type = shared_ctx.m_constants_meta_data[const_src_id_from_blob][const_id_from_blob].m_type;
        // restore weight buffer from file
        auto w_buffer = load_mmap_object(weights_path);
        ASSERT_TRUE(w_buffer);
        // recreate constant buffer with weight_buffer as hint
        constant_buffer = weight_sharing::get_buffer(shared_ctx, w_buffer, const_id_from_blob);
        ASSERT_TRUE(constant_buffer);

        auto c = std::make_shared<Constant>(c_type, shape_from_blob, constant_buffer);
        EXPECT_EQ(c->cast_vector<float>(), ref_data);
        EXPECT_EQ(ov::wsh::Extension::get_constant_source_id(*c), const_src_id_from_blob);
        EXPECT_EQ(ov::wsh::Extension::get_constant_id(*c), const_id_from_blob);
    }
}
}  // namespace ov::test
