// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "lazy_tensor.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"
#include "weights_bank.hpp"

namespace {

using ov::npuw::weights::Bank;
using ov::npuw::weights::LazyTensor;

std::vector<uint8_t> make_bytes(const ov::Shape& shape) {
    std::vector<uint8_t> bytes(ov::shape_size(shape));
    std::iota(bytes.begin(), bytes.end(), uint8_t{1});
    return bytes;
}

// Owns a fresh buffer
std::shared_ptr<ov::op::v0::Constant> make_u8_const(const ov::Shape& shape) {
    return std::make_shared<ov::op::v0::Constant>(ov::element::u8, shape, make_bytes(shape));
}

// Same buffer, different shape - what group merging leaves behind
std::shared_ptr<ov::op::v0::Constant> reshape_const(const std::shared_ptr<ov::op::v0::Constant>& source,
                                                    const ov::Shape& shape) {
    return std::make_shared<ov::op::v0::Constant>(*source, shape);
}

class CountingRemoteContext final : public ov::IRemoteContext {
public:
    const std::string& get_device_name() const override {
        return m_name;
    }
    const ov::AnyMap& get_property() const override {
        return m_properties;
    }
    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type&, const ov::Shape&, const ov::AnyMap&) override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    ov::SoPtr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape) override {
        if (m_failures_left > 0) {
            --m_failures_left;
            OPENVINO_THROW("injected allocation failure");
        }
        ++m_host_allocations;
        return ov::SoPtr<ov::ITensor>(ov::make_tensor(type, shape), nullptr);
    }

    // The next `count` allocations throw; they are not counted as allocations
    void fail_next_allocations(std::size_t count) {
        m_failures_left = count;
    }

    std::size_t host_allocations() const {
        return m_host_allocations;
    }

private:
    std::string m_name = "NPU";
    ov::AnyMap m_properties;
    std::size_t m_host_allocations = 0;
    std::size_t m_failures_left = 0;
};

}  // namespace

TEST(LazyTensorConstDataPtrTest, ReshapedViewsOfOneBufferShareThePointer) {
    const auto base = make_u8_const(ov::Shape{4, 8});
    const auto view = reshape_const(base, ov::Shape{2, 16});

    const auto* base_ptr = LazyTensor(base).const_data_ptr();
    ASSERT_NE(base_ptr, nullptr);
    EXPECT_EQ(base_ptr, LazyTensor(view).const_data_ptr());
    // The tensors themselves stay distinct, which is why the bank has to alias them explicitly
    EXPECT_NE(LazyTensor(base), LazyTensor(view));
}

TEST(LazyTensorConstDataPtrTest, SeparateBuffersHaveDifferentPointers) {
    const auto first = make_u8_const(ov::Shape{4, 8});
    const auto second = make_u8_const(ov::Shape{4, 8});

    EXPECT_NE(LazyTensor(first).const_data_ptr(), LazyTensor(second).const_data_ptr());
}

TEST(LazyTensorConstDataPtrTest, HasNoPointerForEmptyTransformedOrDetachedTensors) {
    EXPECT_EQ(LazyTensor().const_data_ptr(), nullptr);

    const auto base =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2, 4}, std::vector<float>(8, 1.0f));
    EXPECT_EQ(LazyTensor(base).convert(ov::element::f16).const_data_ptr(), nullptr);

    LazyTensor detached(make_u8_const(ov::Shape{4, 8}));
    ASSERT_NE(detached.const_data_ptr(), nullptr);
    detached.detach();
    EXPECT_EQ(detached.const_data_ptr(), nullptr);
}

class WeightsBankAliasTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_context = std::make_shared<CountingRemoteContext>();
        auto core = std::make_shared<testing::NiceMock<ov::MockICore>>();
        ON_CALL(*core, get_default_context(testing::_))
            .WillByDefault(testing::Return(ov::SoPtr<ov::IRemoteContext>(m_context, nullptr)));
        // Empty allocation device: every tensor is allocated on the device it was registered for
        m_bank = std::make_shared<Bank>(core, "", "weights_bank_alias_test");
    }

    std::shared_ptr<CountingRemoteContext> m_context;
    std::shared_ptr<Bank> m_bank;
};

TEST_F(WeightsBankAliasTest, ReshapedViewsShareOneAllocation) {
    const auto base = make_u8_const(ov::Shape{4, 8});
    const auto expected = make_bytes(ov::Shape{4, 8});

    const auto base_uid = m_bank->registerLT(LazyTensor(base), "NPU");
    const auto view_uid = m_bank->registerLT(LazyTensor(reshape_const(base, ov::Shape{2, 16})), "NPU");
    const auto flat_uid = m_bank->registerLT(LazyTensor(reshape_const(base, ov::Shape{32})), "NPU");
    ASSERT_NE(base_uid, view_uid);
    ASSERT_NE(base_uid, flat_uid);

    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 1u);

    auto base_tensor = m_bank->get(base_uid, "NPU");
    auto view_tensor = m_bank->get(view_uid, "NPU");
    auto flat_tensor = m_bank->get(flat_uid, "NPU");

    EXPECT_EQ(base_tensor.get_shape(), (ov::Shape{4, 8}));
    EXPECT_EQ(view_tensor.get_shape(), (ov::Shape{2, 16}));
    EXPECT_EQ(flat_tensor.get_shape(), (ov::Shape{32}));
    EXPECT_EQ(base_tensor.get_element_type(), ov::element::u8);
    EXPECT_EQ(view_tensor.get_element_type(), ov::element::u8);

    EXPECT_EQ(view_tensor.data(), base_tensor.data());
    EXPECT_EQ(flat_tensor.data(), base_tensor.data());
    EXPECT_EQ(std::memcmp(view_tensor.data(), expected.data(), expected.size()), 0);
    EXPECT_EQ(std::memcmp(flat_tensor.data(), expected.data(), expected.size()), 0);
}

TEST_F(WeightsBankAliasTest, SeparateBuffersAreAllocatedSeparately) {
    // Equal shape and contents, but not the same buffer: nothing to share
    const auto first_uid = m_bank->registerLT(LazyTensor(make_u8_const(ov::Shape{4, 8})), "NPU");
    const auto second_uid = m_bank->registerLT(LazyTensor(make_u8_const(ov::Shape{4, 8})), "NPU");
    ASSERT_NE(first_uid, second_uid);

    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 2u);
    EXPECT_NE(m_bank->get(first_uid, "NPU").data(), m_bank->get(second_uid, "NPU").data());
}

TEST_F(WeightsBankAliasTest, SameAddressWithAnotherTypeOrSizeIsAllocatedSeparately) {
    auto buffer = std::make_shared<ov::AlignedBuffer>(32);
    std::memset(buffer->get_ptr(), 7, 32);
    const auto u8_full = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{32}, buffer);
    const auto i8_full = std::make_shared<ov::op::v0::Constant>(ov::element::i8, ov::Shape{32}, buffer);
    // A Constant needs a buffer of its exact size: wrap the first half of the same memory
    auto half = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(buffer->get_ptr<char>(),
                                                                                        16,
                                                                                        buffer);
    const auto u8_half = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{16}, half);
    ASSERT_EQ(LazyTensor(u8_full).const_data_ptr(), LazyTensor(i8_full).const_data_ptr());
    ASSERT_EQ(LazyTensor(u8_full).const_data_ptr(), LazyTensor(u8_half).const_data_ptr());

    const auto u8_uid = m_bank->registerLT(LazyTensor(u8_full), "NPU");
    const auto i8_uid = m_bank->registerLT(LazyTensor(i8_full), "NPU");
    const auto half_uid = m_bank->registerLT(LazyTensor(u8_half), "NPU");

    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 3u);
    EXPECT_EQ(m_bank->get(i8_uid, "NPU").get_element_type(), ov::element::i8);
    EXPECT_EQ(m_bank->get(half_uid, "NPU").get_byte_size(), 16u);
    EXPECT_NE(m_bank->get(u8_uid, "NPU").data(), m_bank->get(i8_uid, "NPU").data());
    EXPECT_NE(m_bank->get(u8_uid, "NPU").data(), m_bank->get(half_uid, "NPU").data());
}

TEST_F(WeightsBankAliasTest, TransformedTensorsAreNotAliasedToTheirSource) {
    const auto base =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2, 4}, std::vector<float>(8, 1.0f));

    const auto plain_uid = m_bank->registerLT(LazyTensor(base), "NPU");
    const auto converted_uid = m_bank->registerLT(LazyTensor(base).convert(ov::element::f16), "NPU");
    ASSERT_NE(plain_uid, converted_uid);

    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 2u);
    EXPECT_EQ(m_bank->get(plain_uid, "NPU").get_element_type(), ov::element::f32);
    EXPECT_EQ(m_bank->get(converted_uid, "NPU").get_element_type(), ov::element::f16);
}

TEST_F(WeightsBankAliasTest, ViewsRegisteredInALaterPassShareTheEarlierAllocation) {
    // Prefill and kvcache models are finalized in separate passes on one bank
    const auto base = make_u8_const(ov::Shape{4, 8});
    const auto expected = make_bytes(ov::Shape{4, 8});
    const auto base_uid = m_bank->registerLT(LazyTensor(base), "NPU");
    m_bank->evaluate_and_allocate();
    ASSERT_EQ(m_context->host_allocations(), 1u);

    const auto view_uid = m_bank->registerLT(LazyTensor(reshape_const(base, ov::Shape{2, 16})), "NPU");
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 1u);
    auto view_tensor = m_bank->get(view_uid, "NPU");
    EXPECT_EQ(view_tensor.data(), m_bank->get(base_uid, "NPU").data());
    EXPECT_EQ(view_tensor.get_shape(), (ov::Shape{2, 16}));
    EXPECT_EQ(std::memcmp(view_tensor.data(), expected.data(), expected.size()), 0);
}

TEST_F(WeightsBankAliasTest, StaleEarlierOwnerIsNotReusedWhenTheSourceAddressHoldsOtherData) {
    const auto base = make_u8_const(ov::Shape{4, 8});
    const auto base_uid = m_bank->registerLT(LazyTensor(base), "NPU");
    m_bank->evaluate_and_allocate();
    ASSERT_EQ(m_context->host_allocations(), 1u);

    // Same address, type and size, new contents: what a freed and reallocated source buffer looks like
    auto* raw = static_cast<uint8_t*>(const_cast<void*>(base->get_data_ptr()));
    std::fill(raw, raw + 32, uint8_t{0xAB});
    const auto view_uid = m_bank->registerLT(LazyTensor(reshape_const(base, ov::Shape{2, 16})), "NPU");
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 2u);
    auto view_tensor = m_bank->get(view_uid, "NPU");
    EXPECT_NE(view_tensor.data(), m_bank->get(base_uid, "NPU").data());
    EXPECT_EQ(static_cast<const uint8_t*>(view_tensor.data())[0], 0xAB);
}

TEST_F(WeightsBankAliasTest, BankRecoversWhenTheFirstAllocationThrows) {
    const auto base = make_u8_const(ov::Shape{4, 8});
    const auto expected = make_bytes(ov::Shape{4, 8});
    const auto base_uid = m_bank->registerLT(LazyTensor(base), "NPU");
    const auto view_uid = m_bank->registerLT(LazyTensor(reshape_const(base, ov::Shape{2, 16})), "NPU");

    m_context->fail_next_allocations(1);
    EXPECT_THROW(m_bank->evaluate_and_allocate(), ov::Exception);
    EXPECT_EQ(m_context->host_allocations(), 0u);

    // The owner must be allocated on retry, not mistaken for a view of itself
    ASSERT_NO_THROW(m_bank->evaluate_and_allocate());

    EXPECT_EQ(m_context->host_allocations(), 1u);
    auto base_tensor = m_bank->get(base_uid, "NPU");
    auto view_tensor = m_bank->get(view_uid, "NPU");
    EXPECT_EQ(view_tensor.data(), base_tensor.data());
    EXPECT_EQ(view_tensor.get_shape(), (ov::Shape{2, 16}));
    EXPECT_EQ(std::memcmp(base_tensor.data(), expected.data(), expected.size()), 0);
}
