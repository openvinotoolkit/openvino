// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "intel_npu/utils/stable_tensor_view.hpp"
#include "lazy_tensor.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
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

struct EmbeddedConstPayload {
    std::string type;
    ov::Shape shape;
    std::size_t offset = 0;
    std::size_t byte_size;
    bool contains_weight = true;
    ov::Tensor weight;

    void serialize(ov::npuw::orc::Stream& stream) {
        stream & type & shape & offset & byte_size & contains_weight;
        if (contains_weight) {
            stream & weight;
        }
    }
};

struct ReshapePayload {
    EmbeddedConstPayload source;
    ov::Shape shape;
    bool canonical_offset = false;

    void serialize(ov::npuw::orc::Stream& stream) {
        const bool initialized = true;
        const std::size_t hash = 42;
        auto source_section = ov::npuw::orc::make_payload_section(1, 0, source);
        stream & initialized & hash & source_section & shape & canonical_offset;
    }
};

std::string serialize_embedded_const(const EmbeddedConstPayload& source, bool reshape) {
    const bool initialized = true;
    const std::size_t hash = 42;
    auto shape = source.shape;
    shape.insert(shape.begin(), 1);
    auto section = reshape ? ov::npuw::orc::make_payload_section(8, 0, ReshapePayload{source, shape})
                           : ov::npuw::orc::make_payload_section(1, 0, source);
    std::stringstream output;
    auto stream = ov::npuw::orc::Stream::writer(output);
    stream & initialized & hash & section;
    return output.str();
}

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

TEST(LazyTensorReshapeTest, ReshapePreservesDataAndRejectsDifferentElementCounts) {
    const auto base = make_u8_const(ov::Shape{4, 8});
    const LazyTensor source(base);
    auto reshaped = source.reshape(ov::Shape{2, 16});
    auto reshaped_again = source.reshape(ov::Shape{2, 16});
    EXPECT_EQ(reshaped, reshaped_again);
    EXPECT_EQ(reshaped.get_hash(), reshaped_again.get_hash());
    EXPECT_NE(reshaped, source.reshape(ov::Shape{8, 4}));
    EXPECT_THROW(source.reshape(ov::Shape{31}), ov::Exception);

    auto result = reshaped.eval();
    EXPECT_EQ(result.get_shape(), (ov::Shape{2, 16}));
    EXPECT_EQ(result.get_element_type(), ov::element::u8);
    const auto expected = make_bytes(ov::Shape{4, 8});
    EXPECT_EQ(std::memcmp(result.data(), expected.data(), expected.size()), 0);
}

class WeightsBankAliasTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_context = std::make_shared<CountingRemoteContext>();
        m_core = std::make_shared<testing::NiceMock<ov::MockICore>>();
        ON_CALL(*m_core, get_default_context(testing::_))
            .WillByDefault(testing::Return(ov::SoPtr<ov::IRemoteContext>(m_context, nullptr)));
        // Empty allocation device: every tensor is allocated on the device it was registered for
        m_bank = std::make_shared<Bank>(m_core, "", "weights_bank_alias_test");
    }

    std::shared_ptr<CountingRemoteContext> m_context;
    std::shared_ptr<testing::NiceMock<ov::MockICore>> m_core;
    std::shared_ptr<Bank> m_bank;
};

TEST_F(WeightsBankAliasTest, RejectMalformedEmbeddedConstantsBeforeRegistration) {
    struct TestCase {
        const char* name;
        EmbeddedConstPayload payload;
        const char* error;
    };

    const std::vector<TestCase> cases = {
        {"short payload",
         {"u8", {1024}, 0, 1024, true, ov::Tensor(ov::element::u8, {1})},
         "embedded weight shape does not match its metadata"},
        {"different shape",
         {"u8", {2, 2}, 0, 4, true, ov::Tensor(ov::element::u8, {4})},
         "embedded weight shape does not match its metadata"},
        {"different type",
         {"u8", {4}, 0, 4, true, ov::Tensor(ov::element::i8, {4})},
         "embedded weight type does not match its metadata"},
        {"different byte size",
         {"u8", {4}, 0, 5, true, ov::Tensor(ov::element::u8, {4})},
         "embedded weight byte size does not match its type and shape"},
        {"short packed payload",
         {"u4", {3}, 0, 2, true, ov::Tensor(ov::element::u4, {1})},
         "embedded weight shape does not match its metadata"},
        {"incorrect packed byte size",
         {"u4", {3}, 0, 1, true, ov::Tensor(ov::element::u4, {3})},
         "embedded weight byte size does not match its type and shape"},
        {"dynamic type",
         {"dynamic", {4}, 0, 4, true, ov::Tensor(ov::element::u8, {4})},
         "embedded weight has unsupported element type"},
        {"shape overflow",
         {"u8", {std::numeric_limits<std::size_t>::max(), 2}, 0, 0, true, ov::Tensor(ov::element::u8, {1})},
         "embedded weight byte size does not match its type and shape"},
        {"uninitialized payload", {"u8", {4}, 0, 4, true, ov::Tensor()}, "embedded weight is not initialized"},
    };

    for (const bool reshape : {false, true}) {
        for (const auto& test : cases) {
            SCOPED_TRACE(test.name);
            SCOPED_TRACE(reshape ? "Reshape source" : "Const root");
            std::stringstream input(serialize_embedded_const(test.payload, reshape));
            LazyTensor imported;
            try {
                ov::npuw::s11n::read(input, imported);
                ADD_FAILURE() << "Malformed embedded weight was accepted";
            } catch (const ov::Exception& ex) {
                EXPECT_THAT(std::string(ex.what()), testing::HasSubstr(test.error));
            }
        }
    }
    EXPECT_EQ(m_context->host_allocations(), 0u);
}

TEST_F(WeightsBankAliasTest, EmbeddedPackedWeightReshapeRemainsShareable) {
    ov::Tensor weight(ov::element::u4, {3});
    std::memset(weight.data(), 0x21, weight.get_byte_size());
    const EmbeddedConstPayload payload{"u4", {3}, 0, 2, true, weight};
    std::stringstream source_input(serialize_embedded_const(payload, false));
    std::stringstream reshape_input(serialize_embedded_const(payload, true));
    LazyTensor source;
    LazyTensor reshaped;
    ASSERT_NO_THROW(ov::npuw::s11n::read(source_input, source));
    ASSERT_NO_THROW(ov::npuw::s11n::read(reshape_input, reshaped));

    const auto source_uid = m_bank->registerLT(source, "NPU");
    const auto reshape_uid = m_bank->registerLT(reshaped, "NPU");
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 1u);
    EXPECT_EQ(m_bank->get(source_uid, "NPU").data(), m_bank->get(reshape_uid, "NPU").data());
    EXPECT_EQ(m_bank->get(reshape_uid, "NPU").get_byte_size(), 2u);
    EXPECT_EQ(std::memcmp(m_bank->get(reshape_uid, "NPU").data(), weight.data(), 2), 0);
}

TEST(WeightsBankAliasSerializationTest, ValidEmptyEmbeddedWeightsAreAccepted) {
    const EmbeddedConstPayload payload{"u8", {0}, 0, 0, true, ov::Tensor(ov::element::u8, {0})};
    for (const bool reshape : {false, true}) {
        std::stringstream input(serialize_embedded_const(payload, reshape));
        LazyTensor imported;
        ASSERT_NO_THROW(ov::npuw::s11n::read(input, imported));
        const auto evaluated = imported.eval();
        EXPECT_EQ(evaluated.get_element_type(), ov::element::u8);
        EXPECT_EQ(evaluated.get_byte_size(), 0u);
        EXPECT_EQ(evaluated.get_shape(), reshape ? (ov::Shape{1, 0}) : (ov::Shape{0}));
    }
}

TEST_F(WeightsBankAliasTest, RepeatedRegistrationOfTheSameLazyTensorRemainsEvaluable) {
    const LazyTensor tensor(make_u8_const(ov::Shape{4, 8}));
    const auto first_uid = m_bank->registerLT(tensor, "NPU");
    const auto second_uid = m_bank->registerLT(tensor, "NPU");
    ASSERT_EQ(first_uid, second_uid);
    ASSERT_NO_THROW(m_bank->evaluate_and_allocate());
    EXPECT_EQ(m_context->host_allocations(), 1u);
    EXPECT_EQ(static_cast<const uint8_t*>(m_bank->get(first_uid, "NPU").data())[0], 1);
}

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
    const auto view_impl =
        std::dynamic_pointer_cast<intel_npu::StableTensorView>(ov::get_tensor_impl(view_tensor)._ptr);
    ASSERT_TRUE(view_impl);
    EXPECT_EQ(view_impl->owner()->data(), base_tensor.data());
}

TEST_F(WeightsBankAliasTest, ExplicitReshapesShareOneAllocation) {
    const auto base = make_u8_const(ov::Shape{4, 8});
    const auto first = LazyTensor(base).canonicalize_constant_shape();
    const auto second = LazyTensor(reshape_const(base, ov::Shape{2, 16})).canonicalize_constant_shape();
    const auto first_uid = m_bank->registerLT(first, "NPU");
    const auto second_uid = m_bank->registerLT(second, "NPU");

    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 1u);
    EXPECT_EQ(m_bank->get(first_uid, "NPU").data(), m_bank->get(second_uid, "NPU").data());
    EXPECT_EQ(m_bank->get(first_uid, "NPU").get_shape(), (ov::Shape{4, 8}));
    EXPECT_EQ(m_bank->get(second_uid, "NPU").get_shape(), (ov::Shape{2, 16}));
}

TEST_F(WeightsBankAliasTest, ExplicitReshapesShareAcrossAllocationPasses) {
    const auto base = make_u8_const(ov::Shape{4, 8});
    const auto first_uid = m_bank->registerLT(LazyTensor(base).canonicalize_constant_shape(), "NPU");
    m_bank->evaluate_and_allocate();
    ASSERT_EQ(m_context->host_allocations(), 1u);

    const auto second_uid =
        m_bank->registerLT(LazyTensor(reshape_const(base, ov::Shape{2, 16})).canonicalize_constant_shape(), "NPU");
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 1u);
    EXPECT_EQ(m_bank->get(first_uid, "NPU").data(), m_bank->get(second_uid, "NPU").data());
}

TEST_F(WeightsBankAliasTest, SameWeightlessOffsetSharesDifferentSourceAddresses) {
    const auto first = make_u8_const(ov::Shape{4, 8});
    const auto second = make_u8_const(ov::Shape{2, 16});
    ASSERT_NE(first->get_data_ptr(), second->get_data_ptr());
    for (const auto& constant : {first, second}) {
        constant->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
            ov::WeightlessCacheAttribute(32, 128, ov::element::u8);
    }
    const auto first_uid = m_bank->registerLT(LazyTensor(first).canonicalize_constant_shape(), "NPU");
    const auto second_uid = m_bank->registerLT(LazyTensor(second).canonicalize_constant_shape(), "NPU");
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 1u);
    EXPECT_EQ(m_bank->get(first_uid, "NPU").data(), m_bank->get(second_uid, "NPU").data());
}

TEST_F(WeightsBankAliasTest, DifferentBytesAtSameWeightlessOffsetRemainSeparate) {
    const auto first = make_u8_const(ov::Shape{4, 8});
    const auto second = make_u8_const(ov::Shape{2, 16});
    auto* raw = static_cast<uint8_t*>(const_cast<void*>(second->get_data_ptr()));
    std::fill(raw, raw + 32, uint8_t{0xAB});
    for (const auto& constant : {first, second}) {
        constant->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
            ov::WeightlessCacheAttribute(32, 128, ov::element::u8);
    }
    const auto first_uid = m_bank->registerLT(LazyTensor(first).canonicalize_constant_shape(), "NPU");
    const auto second_uid = m_bank->registerLT(LazyTensor(second).canonicalize_constant_shape(), "NPU");
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 2u);
    EXPECT_NE(m_bank->get(first_uid, "NPU").data(), m_bank->get(second_uid, "NPU").data());
    EXPECT_EQ(static_cast<const uint8_t*>(m_bank->get(first_uid, "NPU").data())[0], 1);
    EXPECT_EQ(static_cast<const uint8_t*>(m_bank->get(second_uid, "NPU").data())[0], 0xAB);
}

TEST_F(WeightsBankAliasTest, ChangedWeightlessOffsetContentsAreNotReusedAcrossPasses) {
    const auto first = make_u8_const(ov::Shape{4, 8});
    first->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute(32, 128, ov::element::u8);
    const auto first_uid = m_bank->registerLT(LazyTensor(first).canonicalize_constant_shape(), "NPU");
    m_bank->evaluate_and_allocate();

    auto* raw = static_cast<uint8_t*>(const_cast<void*>(first->get_data_ptr()));
    std::fill(raw, raw + 32, uint8_t{0xAB});
    const auto second = reshape_const(first, ov::Shape{2, 16});
    second->get_rt_info() = first->get_rt_info();
    const auto second_uid = m_bank->registerLT(LazyTensor(second).canonicalize_constant_shape(), "NPU");
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 2u);
    EXPECT_NE(m_bank->get(first_uid, "NPU").data(), m_bank->get(second_uid, "NPU").data());
    EXPECT_EQ(static_cast<const uint8_t*>(m_bank->get(first_uid, "NPU").data())[0], 1);
    EXPECT_EQ(static_cast<const uint8_t*>(m_bank->get(second_uid, "NPU").data())[0], 0xAB);
}

TEST_F(WeightsBankAliasTest, WeightlessReshapesShareAfterSerialization) {
    const auto base = make_u8_const(ov::Shape{4, 8});
    base->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute(base->get_byte_size(), 0, ov::element::u8);
    const auto second_const = make_u8_const(ov::Shape{2, 16});
    ASSERT_NE(base->get_data_ptr(), second_const->get_data_ptr());
    second_const->get_rt_info() = base->get_rt_info();
    auto first = LazyTensor(base).canonicalize_constant_shape();
    auto second = LazyTensor(second_const).canonicalize_constant_shape();

    std::stringstream serialized;
    ov::npuw::s11n::write(serialized, first);
    ov::npuw::s11n::write(serialized, second);
    LazyTensor imported_first;
    LazyTensor imported_second;
    ov::npuw::s11n::read(serialized, imported_first);
    ov::npuw::s11n::read(serialized, imported_second);
    ov::npuw::s11n::WeightsContext::ConstsCache cache;
    cache[{0, base->get_byte_size()}] = base;
    const ov::npuw::s11n::WeightsContext ctx(nullptr, "", cache, {});
    imported_first.read_weight(ctx);
    imported_second.read_weight(ctx);

    const auto first_uid = m_bank->registerLT(imported_first, "NPU");
    m_bank->evaluate_and_allocate();
    const auto second_uid = m_bank->registerLT(imported_second, "NPU");
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 1u);
    EXPECT_EQ(m_bank->get(first_uid, "NPU").data(), m_bank->get(second_uid, "NPU").data());
    EXPECT_EQ(m_bank->get(first_uid, "NPU").get_shape(), (ov::Shape{4, 8}));
    EXPECT_EQ(m_bank->get(second_uid, "NPU").get_shape(), (ov::Shape{2, 16}));
}

TEST_F(WeightsBankAliasTest, WeightlessCacheSupportsDifferentTypesAtOneOffset) {
    auto buffer = std::make_shared<ov::AlignedBuffer>(32);
    std::memset(buffer->get_ptr(), 42, 32);
    const auto u8 = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{32}, buffer);
    const auto i8 = std::make_shared<ov::op::v0::Constant>(ov::element::i8, ov::Shape{4, 8}, buffer);
    for (const auto& constant : {u8, i8}) {
        constant->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
            ov::WeightlessCacheAttribute(32, 0, constant->get_element_type());
    }
    auto first = LazyTensor(u8).canonicalize_constant_shape();
    auto second = LazyTensor(i8).canonicalize_constant_shape();
    std::stringstream serialized;
    ov::npuw::s11n::write(serialized, first);
    ov::npuw::s11n::write(serialized, second);
    LazyTensor imported_first;
    LazyTensor imported_second;
    ov::npuw::s11n::read(serialized, imported_first);
    ov::npuw::s11n::read(serialized, imported_second);
    ov::npuw::s11n::WeightsContext::ConstsCache cache;
    cache[{0, 32}] = i8;
    const ov::npuw::s11n::WeightsContext ctx(nullptr, "", cache, {});
    imported_first.read_weight(ctx);
    imported_second.read_weight(ctx);
    const auto first_uid = m_bank->registerLT(imported_first, "NPU");
    const auto second_uid = m_bank->registerLT(imported_second, "NPU");
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 2u);
    EXPECT_EQ(m_bank->get(first_uid, "NPU").get_element_type(), ov::element::u8);
    EXPECT_EQ(m_bank->get(second_uid, "NPU").get_element_type(), ov::element::i8);
    EXPECT_EQ(static_cast<const uint8_t*>(m_bank->get(first_uid, "NPU").data())[0], 42);
}

TEST_F(WeightsBankAliasTest, ImportedReshapeRejectsReusedSourceIdentityWithDifferentData) {
    const auto base = make_u8_const(ov::Shape{4, 8});
    auto first = LazyTensor(base).canonicalize_constant_shape();
    std::stringstream serialized;
    ov::npuw::s11n::write(serialized, first);
    auto* raw = static_cast<uint8_t*>(const_cast<void*>(base->get_data_ptr()));
    std::fill(raw, raw + 32, uint8_t{0xAB});
    auto second = LazyTensor(reshape_const(base, ov::Shape{2, 16})).canonicalize_constant_shape();
    ov::npuw::s11n::write(serialized, second);

    LazyTensor imported_first;
    LazyTensor imported_second;
    ov::npuw::s11n::read(serialized, imported_first);
    ov::npuw::s11n::read(serialized, imported_second);
    const auto first_uid = m_bank->registerLT(imported_first, "NPU");
    m_bank->evaluate_and_allocate();
    const auto second_uid = m_bank->registerLT(imported_second, "NPU");
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 2u);
    EXPECT_NE(m_bank->get(first_uid, "NPU").data(), m_bank->get(second_uid, "NPU").data());
    EXPECT_EQ(static_cast<const uint8_t*>(m_bank->get(first_uid, "NPU").data())[0], 1);
    EXPECT_EQ(static_cast<const uint8_t*>(m_bank->get(second_uid, "NPU").data())[0], 0xAB);
}

TEST_F(WeightsBankAliasTest, EmbeddedSourceCanBeSerializedForMultipleReshapes) {
    const LazyTensor source(make_u8_const(ov::Shape{32}));
    auto first = source.reshape(ov::Shape{4, 8});
    auto second = source.reshape(ov::Shape{2, 16});
    std::stringstream serialized;
    ov::npuw::s11n::write(serialized, first);
    ov::npuw::s11n::write(serialized, second);

    LazyTensor imported_first;
    LazyTensor imported_second;
    ov::npuw::s11n::read(serialized, imported_first);
    ov::npuw::s11n::read(serialized, imported_second);
    const auto first_uid = m_bank->registerLT(imported_first, "NPU");
    const auto second_uid = m_bank->registerLT(imported_second, "NPU");
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 1u);
    EXPECT_EQ(m_bank->get(first_uid, "NPU").data(), m_bank->get(second_uid, "NPU").data());
    EXPECT_EQ(static_cast<const uint8_t*>(m_bank->get(second_uid, "NPU").data())[0], 1);
}

TEST_F(WeightsBankAliasTest, FullBankRoundTripPreservesReshapedViews) {
    const auto base = make_u8_const(ov::Shape{4, 8});
    const auto first_uid = m_bank->registerLT(LazyTensor(base).canonicalize_constant_shape(), "NPU");
    const auto second_uid =
        m_bank->registerLT(LazyTensor(reshape_const(base, ov::Shape{2, 16})).canonicalize_constant_shape(), "NPU");
    m_bank->evaluate_and_allocate();
    ASSERT_EQ(m_context->host_allocations(), 1u);

    std::stringstream serialized;
    ov::npuw::s11n::write(serialized, *m_bank);
    Bank restored(m_core, "", "restored_weights_bank");
    ov::npuw::s11n::read(serialized, restored);

    EXPECT_EQ(m_context->host_allocations(), 2u);
    const auto first = restored.get(first_uid, "NPU");
    const auto second = restored.get(second_uid, "NPU");
    EXPECT_EQ(first.data(), second.data());
    EXPECT_EQ(first.get_shape(), (ov::Shape{4, 8}));
    EXPECT_EQ(second.get_shape(), (ov::Shape{2, 16}));
    const auto expected = make_bytes(ov::Shape{4, 8});
    EXPECT_EQ(std::memcmp(second.data(), expected.data(), expected.size()), 0);
}

TEST_F(WeightsBankAliasTest, LegacyFullBankStillImports) {
    std::stringstream serialized;
    auto stream = ov::npuw::orc::Stream::writer(serialized);
    const std::size_t bank_size = 1;
    const std::size_t storage_size = 2;
    const std::string device = "NPU";
    const int64_t first_uid = 0;
    const int64_t second_uid = 1;
    auto first = ov::Tensor(ov::element::u8, ov::Shape{4, 8});
    auto second = ov::Tensor(ov::element::u8, ov::Shape{2, 16});
    std::memset(first.data(), 7, first.get_byte_size());
    std::memset(second.data(), 9, second.get_byte_size());
    stream & bank_size & device & storage_size & first_uid;
    ov::npuw::orc::transfer_tensor(stream, first);
    stream & second_uid;
    ov::npuw::orc::transfer_tensor(stream, second);

    Bank restored(m_core, "", "legacy_weights_bank");
    ov::npuw::s11n::read(serialized, restored);

    EXPECT_EQ(m_context->host_allocations(), 2u);
    EXPECT_EQ(static_cast<const uint8_t*>(restored.get(first_uid, "NPU").data())[0], 7);
    EXPECT_EQ(static_cast<const uint8_t*>(restored.get(second_uid, "NPU").data())[0], 9);
}

TEST_F(WeightsBankAliasTest, FullBankRejectsInvalidViewOwnerAndShape) {
    for (const bool invalid_owner : {true, false}) {
        std::stringstream serialized;
        auto stream = ov::npuw::orc::Stream::writer(serialized);
        const std::size_t marker = std::numeric_limits<std::size_t>::max();
        const std::uint16_t version = 1;
        const std::size_t bank_size = 1;
        const std::size_t storage_size = 2;
        const std::string device = "NPU";
        const int64_t first_uid = 0;
        const int64_t second_uid = 1;
        const bool owner_is_view = false;
        const bool is_view = true;
        const int64_t owner_uid = invalid_owner ? 2 : 0;
        const ov::Shape bad_shape{31};
        auto owner = ov::Tensor(ov::element::u8, ov::Shape{32});
        stream & marker & version & bank_size & device & storage_size & first_uid & owner_is_view;
        ov::npuw::orc::transfer_tensor(stream, owner);
        stream & second_uid & is_view & owner_uid & bad_shape;
        Bank restored(m_core, "", "invalid_view_bank");
        EXPECT_THROW(ov::npuw::s11n::read(serialized, restored), ov::Exception);
    }
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
    auto half =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(buffer->get_ptr<char>(), 16, buffer);
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

TEST_F(WeightsBankAliasTest, SameAddressWithMultipleLayoutsSharesEachLayoutInOnePass) {
    auto buffer = std::make_shared<ov::AlignedBuffer>(32);
    std::memset(buffer->get_ptr(), 7, 32);
    const auto u8_full = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{32}, buffer);
    const auto i8_full = std::make_shared<ov::op::v0::Constant>(ov::element::i8, ov::Shape{4, 8}, buffer);
    auto half =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(buffer->get_ptr<char>(), 16, buffer);
    const auto u8_half = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{16}, half);
    const auto u8_uid = m_bank->registerLT(LazyTensor(u8_full), "NPU");
    const auto i8_uid = m_bank->registerLT(LazyTensor(i8_full), "NPU");
    const auto half_uid = m_bank->registerLT(LazyTensor(u8_half), "NPU");
    const auto i8_view_uid = m_bank->registerLT(LazyTensor(reshape_const(i8_full, ov::Shape{2, 16})), "NPU");
    const auto half_view_uid = m_bank->registerLT(LazyTensor(reshape_const(u8_half, ov::Shape{4, 4})), "NPU");

    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 3u);
    EXPECT_NE(m_bank->get(u8_uid, "NPU").data(), m_bank->get(i8_uid, "NPU").data());
    EXPECT_NE(m_bank->get(u8_uid, "NPU").data(), m_bank->get(half_uid, "NPU").data());
    EXPECT_EQ(m_bank->get(i8_uid, "NPU").data(), m_bank->get(i8_view_uid, "NPU").data());
    EXPECT_EQ(m_bank->get(half_uid, "NPU").data(), m_bank->get(half_view_uid, "NPU").data());
    EXPECT_EQ(m_bank->get(i8_view_uid, "NPU").get_shape(), (ov::Shape{2, 16}));
    EXPECT_EQ(m_bank->get(half_view_uid, "NPU").get_shape(), (ov::Shape{4, 4}));
}

TEST_F(WeightsBankAliasTest, SameAddressWithMultipleLayoutsSharesEachLayoutAcrossPasses) {
    auto buffer = std::make_shared<ov::AlignedBuffer>(32);
    std::memset(buffer->get_ptr(), 7, 32);
    const auto u8_full = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{32}, buffer);
    const auto i8_full = std::make_shared<ov::op::v0::Constant>(ov::element::i8, ov::Shape{4, 8}, buffer);
    auto half =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(buffer->get_ptr<char>(), 16, buffer);
    const auto u8_half = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{16}, half);
    const auto u8_uid = m_bank->registerLT(LazyTensor(u8_full), "NPU");
    const auto i8_uid = m_bank->registerLT(LazyTensor(i8_full), "NPU");
    const auto half_uid = m_bank->registerLT(LazyTensor(u8_half), "NPU");
    m_bank->evaluate_and_allocate();
    ASSERT_EQ(m_context->host_allocations(), 3u);

    const auto u8_view_uid = m_bank->registerLT(LazyTensor(reshape_const(u8_full, ov::Shape{2, 16})), "NPU");
    const auto i8_view_uid = m_bank->registerLT(LazyTensor(reshape_const(i8_full, ov::Shape{2, 16})), "NPU");
    const auto half_view_uid = m_bank->registerLT(LazyTensor(reshape_const(u8_half, ov::Shape{4, 4})), "NPU");
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 3u);
    EXPECT_EQ(m_bank->get(u8_uid, "NPU").data(), m_bank->get(u8_view_uid, "NPU").data());
    EXPECT_EQ(m_bank->get(i8_uid, "NPU").data(), m_bank->get(i8_view_uid, "NPU").data());
    EXPECT_EQ(m_bank->get(half_uid, "NPU").data(), m_bank->get(half_view_uid, "NPU").data());
}

TEST_F(WeightsBankAliasTest, PackedAndByteLayoutsShareOnlyWithinTheirOwnType) {
    auto buffer = std::make_shared<ov::AlignedBuffer>(32);
    std::memset(buffer->get_ptr(), 0x42, 32);
    const auto u8 = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{32}, buffer);
    const auto u4 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{64}, buffer);
    const auto u8_uid = m_bank->registerLT(LazyTensor(u8), "NPU");
    const auto u4_uid = m_bank->registerLT(LazyTensor(u4), "NPU");
    const auto u4_view_uid = m_bank->registerLT(LazyTensor(reshape_const(u4, ov::Shape{8, 8})), "NPU");

    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 2u);
    EXPECT_NE(m_bank->get(u8_uid, "NPU").data(), m_bank->get(u4_uid, "NPU").data());
    EXPECT_EQ(m_bank->get(u4_uid, "NPU").data(), m_bank->get(u4_view_uid, "NPU").data());
    EXPECT_EQ(m_bank->get(u4_view_uid, "NPU").get_element_type(), ov::element::u4);
    EXPECT_EQ(m_bank->get(u4_view_uid, "NPU").get_byte_size(), 32u);
}

TEST_F(WeightsBankAliasTest, ReplacingStaleOwnerPreservesOtherLayouts) {
    auto buffer = std::make_shared<ov::AlignedBuffer>(32);
    std::memset(buffer->get_ptr(), 7, 32);
    const auto u8 = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{32}, buffer);
    const auto i8 = std::make_shared<ov::op::v0::Constant>(ov::element::i8, ov::Shape{32}, buffer);
    const auto old_u8 = m_bank->registerLT(LazyTensor(u8), "NPU");
    const auto old_i8 = m_bank->registerLT(LazyTensor(i8), "NPU");
    m_bank->evaluate_and_allocate();
    ASSERT_EQ(m_context->host_allocations(), 2u);

    std::memset(buffer->get_ptr(), 42, 32);
    const auto new_i8 = m_bank->registerLT(LazyTensor(reshape_const(i8, ov::Shape{4, 8})), "NPU");
    m_bank->evaluate_and_allocate();
    ASSERT_EQ(m_context->host_allocations(), 3u);
    const auto new_u8 = m_bank->registerLT(LazyTensor(reshape_const(u8, ov::Shape{4, 8})), "NPU");
    m_bank->evaluate_and_allocate();
    ASSERT_EQ(m_context->host_allocations(), 4u);

    const auto i8_view = m_bank->registerLT(LazyTensor(reshape_const(i8, ov::Shape{2, 16})), "NPU");
    const auto u8_view = m_bank->registerLT(LazyTensor(reshape_const(u8, ov::Shape{2, 16})), "NPU");
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 4u);
    EXPECT_NE(m_bank->get(old_u8, "NPU").data(), m_bank->get(new_u8, "NPU").data());
    EXPECT_NE(m_bank->get(old_i8, "NPU").data(), m_bank->get(new_i8, "NPU").data());
    EXPECT_EQ(m_bank->get(i8_view, "NPU").data(), m_bank->get(new_i8, "NPU").data());
    EXPECT_EQ(m_bank->get(u8_view, "NPU").data(), m_bank->get(new_u8, "NPU").data());
    EXPECT_EQ(static_cast<const uint8_t*>(m_bank->get(u8_view, "NPU").data())[0], 42);
}

TEST_F(WeightsBankAliasTest, ViewKeepsItsOwnerAliveAfterBankIsDestroyed) {
    const auto base = make_u8_const(ov::Shape{4, 8});
    const auto base_uid = m_bank->registerLT(LazyTensor(base), "NPU");
    const auto view_uid = m_bank->registerLT(LazyTensor(reshape_const(base, ov::Shape{2, 16})), "NPU");
    m_bank->evaluate_and_allocate();

    auto owner = m_bank->get(base_uid, "NPU");
    auto view = m_bank->get(view_uid, "NPU");
    std::weak_ptr<ov::ITensor> owner_impl = ov::get_tensor_impl(owner)._ptr;
    owner = ov::Tensor();
    m_bank.reset();
    EXPECT_FALSE(owner_impl.expired());
    EXPECT_EQ(static_cast<const uint8_t*>(view.data())[0], 1);
    view = ov::Tensor();
    EXPECT_TRUE(owner_impl.expired());
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

TEST_F(WeightsBankAliasTest, SameShapeWithReusedSourceAddressDoesNotReuseStaleWeights) {
    const auto base = make_u8_const(ov::Shape{4, 8});
    const auto old_uid = m_bank->registerLT(LazyTensor(base), "NPU");
    m_bank->evaluate_and_allocate();
    ASSERT_EQ(m_context->host_allocations(), 1u);

    auto* raw = static_cast<uint8_t*>(const_cast<void*>(base->get_data_ptr()));
    std::fill(raw, raw + 32, uint8_t{0xAB});
    const auto new_uid = m_bank->registerLT(LazyTensor(reshape_const(base, ov::Shape{4, 8})), "NPU");
    ASSERT_NE(old_uid, new_uid);
    m_bank->evaluate_and_allocate();

    EXPECT_EQ(m_context->host_allocations(), 2u);
    EXPECT_NE(m_bank->get(old_uid, "NPU").data(), m_bank->get(new_uid, "NPU").data());
    EXPECT_EQ(static_cast<const uint8_t*>(m_bank->get(old_uid, "NPU").data())[0], 1);
    EXPECT_EQ(static_cast<const uint8_t*>(m_bank->get(new_uid, "NPU").data())[0], 0xAB);
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
