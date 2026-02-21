// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/except.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::test {

// Mock ITensor implementation for testing
class MockTensorImpl : public ov::ITensor {
public:
    MockTensorImpl(const ov::element::Type& element_type, const ov::Shape& shape)
        : m_element_type(element_type),
          m_shape(shape),
          m_byte_size(ov::shape_size(shape) * element_type.size()) {
        if (m_byte_size > 0) {
            m_data = std::malloc(m_byte_size);
            if (!m_data) {
                throw std::bad_alloc();
            }
        }
    }

    ~MockTensorImpl() override {
        if (m_data) {
            std::free(m_data);
        }
    }

    const ov::element::Type& get_element_type() const override {
        return m_element_type;
    }

    void set_shape(ov::Shape new_shape) override {
        m_shape = std::move(new_shape);
    }

    const ov::Shape& get_shape() const override {
        return m_shape;
    }

    size_t get_size() const override {
        return ov::shape_size(m_shape);
    }

    size_t get_byte_size() const override {
        return m_byte_size;
    }

    const ov::Strides& get_strides() const override {
        OPENVINO_THROW("get_strides() not implemented in MockTensorImpl");
    }

    void* data() override {
        return m_data;
    }

    const void* data() const override {
        return m_data;
    }

    void* data_rw() override {
        return m_data;
    }

    void* data(const ov::element::Type& element_type) override {
        if (element_type != m_element_type && element_type.bitwidth() != 0) {
            OPENVINO_THROW("Cannot get data with different element type");
        }
        return m_data;
    }

    const void* data(const ov::element::Type& element_type) const override {
        if (element_type != m_element_type && element_type.bitwidth() != 0) {
            OPENVINO_THROW("Cannot get data with different element type");
        }
        return m_data;
    }

    void* data_rw(const ov::element::Type& element_type) override {
        if (element_type != m_element_type && element_type.bitwidth() != 0) {
            OPENVINO_THROW("Cannot get data with different element type");
        }
        return m_data;
    }

private:
    ov::element::Type m_element_type;
    ov::Shape m_shape;
    size_t m_byte_size;
    void* m_data = nullptr;
};

class OVTensorCustomImplTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset to default state before each test
        ov::util::reset_custom_tensor_impl_generator();
    }

    void TearDown() override {
        // Clean up after each test
        ov::util::reset_custom_tensor_impl_generator();
    }
};

TEST_F(OVTensorCustomImplTest, DefaultBehaviorWithoutCustomGenerator) {
    // Create tensor without custom generator
    ov::Shape shape = {2, 3, 4};
    ov::Tensor tensor(ov::element::f32, shape);

    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_element_type(), ov::element::f32);
    EXPECT_NE(tensor.data(), nullptr);
    EXPECT_EQ(tensor.get_size(), 24);
}

TEST_F(OVTensorCustomImplTest, CustomGeneratorIsUsed) {
    bool generator_called = false;
    ov::Shape captured_shape;
    ov::element::Type captured_type;

    // Set custom generator
    ov::util::set_custom_tensor_impl_generator(
        [&](const ov::element::Type& type, const ov::Shape& shape) -> std::shared_ptr<ov::ITensor> {
            generator_called = true;
            captured_type = type;
            captured_shape = shape;
            return std::make_shared<MockTensorImpl>(type, shape);
        });

    // Create tensor - should use custom generator
    ov::Shape shape = {2, 3, 4};
    ov::Tensor tensor(ov::element::f32, shape);

    EXPECT_TRUE(generator_called);
    EXPECT_EQ(captured_type, ov::element::f32);
    EXPECT_EQ(captured_shape, shape);
    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_element_type(), ov::element::f32);
    EXPECT_NE(tensor.data(), nullptr);
}

TEST_F(OVTensorCustomImplTest, ResetGeneratorRestoresDefaultBehavior) {
    bool generator_called = false;

    // Set custom generator
    ov::util::set_custom_tensor_impl_generator(
        [&](const ov::element::Type& type, const ov::Shape& shape) -> std::shared_ptr<ov::ITensor> {
            generator_called = true;
            return std::make_shared<MockTensorImpl>(type, shape);
        });

    // Create tensor - should use custom generator
    ov::Shape shape1 = {2, 3};
    ov::Tensor tensor1(ov::element::f32, shape1);
    EXPECT_TRUE(generator_called);

    // Reset generator
    generator_called = false;
    ov::util::reset_custom_tensor_impl_generator();

    // Create another tensor - should use default behavior
    ov::Shape shape2 = {4, 5};
    ov::Tensor tensor2(ov::element::f32, shape2);
    EXPECT_FALSE(generator_called);
    EXPECT_EQ(tensor2.get_shape(), shape2);
    EXPECT_NE(tensor2.data(), nullptr);
}

TEST_F(OVTensorCustomImplTest, CustomGeneratorDoesNotAffectHostPtrConstructor) {
    bool generator_called = false;

    // Set custom generator
    ov::util::set_custom_tensor_impl_generator(
        [&](const ov::element::Type& type, const ov::Shape& shape) -> std::shared_ptr<ov::ITensor> {
            generator_called = true;
            return std::make_shared<MockTensorImpl>(type, shape);
        });

    // Create tensor with host pointer - should NOT use custom generator
    ov::Shape shape = {2, 3, 4};
    std::vector<float> data(ov::shape_size(shape), 1.0f);
    ov::Tensor tensor(ov::element::f32, shape, data.data());

    EXPECT_FALSE(generator_called);
    EXPECT_EQ(tensor.data<float>(), data.data());
}

TEST_F(OVTensorCustomImplTest, CustomGeneratorDoesNotAffectCustomAllocator) {
    bool generator_called = false;

    // Set custom generator
    ov::util::set_custom_tensor_impl_generator(
        [&](const ov::element::Type& type, const ov::Shape& shape) -> std::shared_ptr<ov::ITensor> {
            generator_called = true;
            return std::make_shared<MockTensorImpl>(type, shape);
        });

    // Create a simple custom allocator
    struct CustomAllocator {
        void* allocate(size_t bytes, size_t alignment) {
            return alignment == 0 ? std::malloc(bytes) : aligned_alloc(alignment, bytes);
        }
        void deallocate(void* ptr, size_t, size_t) noexcept {
            std::free(ptr);
        }
        bool is_equal(const CustomAllocator&) const noexcept {
            return true;
        }
    };

    CustomAllocator custom_alloc;
    ov::Allocator allocator(custom_alloc);

    // Create tensor with custom allocator - should NOT use custom generator
    ov::Shape shape = {2, 3, 4};
    ov::Tensor tensor(ov::element::f32, shape, allocator);

    EXPECT_FALSE(generator_called);
    EXPECT_NE(tensor.data(), nullptr);
}

TEST_F(OVTensorCustomImplTest, MultipleSetAndReset) {
    bool generator1_called = false;
    bool generator2_called = false;

    // Set first generator
    ov::util::set_custom_tensor_impl_generator(
        [&](const ov::element::Type& type, const ov::Shape& shape) -> std::shared_ptr<ov::ITensor> {
            generator1_called = true;
            return std::make_shared<MockTensorImpl>(type, shape);
        });

    ov::Shape shape = {2, 3};
    ov::Tensor tensor1(ov::element::f32, shape);
    EXPECT_TRUE(generator1_called);
    EXPECT_FALSE(generator2_called);

    // Set second generator (replaces first)
    generator1_called = false;
    ov::util::set_custom_tensor_impl_generator(
        [&](const ov::element::Type& type, const ov::Shape& shape) -> std::shared_ptr<ov::ITensor> {
            generator2_called = true;
            return std::make_shared<MockTensorImpl>(type, shape);
        });

    ov::Tensor tensor2(ov::element::f32, shape);
    EXPECT_FALSE(generator1_called);
    EXPECT_TRUE(generator2_called);

    // Reset and verify default behavior
    ov::util::reset_custom_tensor_impl_generator();
    generator1_called = false;
    generator2_called = false;

    ov::Tensor tensor3(ov::element::f32, shape);
    EXPECT_FALSE(generator1_called);
    EXPECT_FALSE(generator2_called);
}

TEST_F(OVTensorCustomImplTest, CustomGeneratorWithDifferentElementTypes) {
    std::vector<ov::element::Type> captured_types;

    ov::util::set_custom_tensor_impl_generator(
        [&](const ov::element::Type& type, const ov::Shape& shape) -> std::shared_ptr<ov::ITensor> {
            captured_types.push_back(type);
            return std::make_shared<MockTensorImpl>(type, shape);
        });

    ov::Shape shape = {2, 3};
    ov::Tensor tensor_f32(ov::element::f32, shape);
    ov::Tensor tensor_i32(ov::element::i32, shape);
    ov::Tensor tensor_u8(ov::element::u8, shape);

    EXPECT_EQ(captured_types.size(), 3);
    EXPECT_EQ(captured_types[0], ov::element::f32);
    EXPECT_EQ(captured_types[1], ov::element::i32);
    EXPECT_EQ(captured_types[2], ov::element::u8);
}

TEST_F(OVTensorCustomImplTest, CustomGeneratorWithDifferentShapes) {
    std::vector<ov::Shape> captured_shapes;

    ov::util::set_custom_tensor_impl_generator(
        [&](const ov::element::Type& type, const ov::Shape& shape) -> std::shared_ptr<ov::ITensor> {
            captured_shapes.push_back(shape);
            return std::make_shared<MockTensorImpl>(type, shape);
        });

    ov::Tensor tensor1(ov::element::f32, {2, 3});
    ov::Tensor tensor2(ov::element::f32, {4, 5, 6});
    ov::Tensor tensor3(ov::element::f32, {1});

    EXPECT_EQ(captured_shapes.size(), 3);
    EXPECT_EQ(captured_shapes[0], ov::Shape({2, 3}));
    EXPECT_EQ(captured_shapes[1], ov::Shape({4, 5, 6}));
    EXPECT_EQ(captured_shapes[2], ov::Shape({1}));
}

TEST_F(OVTensorCustomImplTest, EmptyShapeTensor) {
    bool generator_called = false;

    ov::util::set_custom_tensor_impl_generator(
        [&](const ov::element::Type& type, const ov::Shape& shape) -> std::shared_ptr<ov::ITensor> {
            generator_called = true;
            return std::make_shared<MockTensorImpl>(type, shape);
        });

    ov::Tensor tensor(ov::element::f32, {0});
    EXPECT_TRUE(generator_called);
    EXPECT_EQ(tensor.get_size(), 0);
}

TEST_F(OVTensorCustomImplTest, SetNullptrGeneratorResetsToDefault) {
    bool generator_called = false;

    // Set a generator
    ov::util::set_custom_tensor_impl_generator(
        [&](const ov::element::Type& type, const ov::Shape& shape) -> std::shared_ptr<ov::ITensor> {
            generator_called = true;
            return std::make_shared<MockTensorImpl>(type, shape);
        });

    ov::Tensor tensor1(ov::element::f32, {2, 3});
    EXPECT_TRUE(generator_called);

    // Set nullptr as generator (should reset to default)
    generator_called = false;
    ov::util::set_custom_tensor_impl_generator(nullptr);

    ov::Tensor tensor2(ov::element::f32, {2, 3});
    EXPECT_FALSE(generator_called);
    EXPECT_NE(tensor2.data(), nullptr);
}

}  // namespace ov::test
