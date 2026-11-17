// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Tests for WeightSharingExtension: conversions of the same GgufDecoder weight share one set of
// page-aligned constant buffers registered for plugin import, and produce the same model as a
// conversion without the extension.

#include <cstring>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "op_test_utils.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/frontend/gguf/extension/weight_sharing.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/util/mmap_object.hpp"

using namespace ov_gguf_test;

namespace {

constexpr size_t kRows = 4;
constexpr size_t kCols = 256;

// Q4_0 blocks (f16 scale + 16 bytes of nibbles per 32 weights) with values that depend on `seed`.
ov::Tensor q4_0_bytes(uint8_t seed) {
    constexpr size_t kBlockBytes = 18;
    ov::Tensor data(ov::element::u8, ov::Shape{kRows * kCols / 32 * kBlockBytes});
    auto* bytes = data.data<uint8_t>();
    for (size_t block = 0; block < kRows * kCols / 32; ++block) {
        const auto scale = ov::float16(0.5f + 0.25f * static_cast<float>((block + seed) % 4)).to_bits();
        std::memcpy(bytes + block * kBlockBytes, &scale, sizeof(scale));
        for (size_t i = 0; i < 16; ++i)
            bytes[block * kBlockBytes + 2 + i] = static_cast<uint8_t>(block * 7 + i * 13 + seed);
    }
    return data;
}

std::shared_ptr<ov::Model> convert_weight(const ov::Tensor& data,
                                          const std::vector<std::shared_ptr<ov::Extension>>& extensions = {}) {
    return SingleOpBuilder()
        .op("GGML_OP_NONE")
        .output("w", ov::element::f32, {kRows, kCols})
        .attr<ov::Tensor>("data", data)
        .attr<std::string>("quant_type", "Q4_0")
        .build_with_extensions(extensions);
}

std::vector<std::shared_ptr<ov::op::v0::Constant>> constants(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<ov::op::v0::Constant>> result;
    for (const auto& node : model->get_ordered_ops())
        if (const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node))
            result.push_back(constant);
    return result;
}

}  // namespace

// Two conversions of one weight read the same buffers, each page-aligned and registered as a
// weight-sharing source.
TEST(GGUFWeightSharing, ConversionsShareConstantBuffers) {
    const auto data = q4_0_bytes(1);
    const auto extension = std::make_shared<ov::frontend::gguf::WeightSharingExtension>();
    const auto first = constants(convert_weight(data, {extension}));
    const auto second = constants(convert_weight(data, {extension}));

    ASSERT_FALSE(first.empty());
    ASSERT_EQ(first.size(), second.size());
    const auto page = static_cast<uintptr_t>(ov::util::get_system_page_size());
    for (size_t i = 0; i < first.size(); ++i) {
        EXPECT_NE(first[i], second[i]) << "each conversion needs its own nodes";
        EXPECT_EQ(first[i]->get_data_ptr(), second[i]->get_data_ptr()) << first[i]->get_friendly_name();
        EXPECT_EQ(reinterpret_cast<uintptr_t>(first[i]->get_data_ptr()) % page, 0u) << first[i]->get_friendly_name();
        EXPECT_NE(ov::weight_sharing::Extension::get_constant_source_id(*first[i]),
                  ov::weight_sharing::invalid_source_id)
            << first[i]->get_friendly_name();
    }
}

// Sharing changes where the constants live, not the model.
TEST(GGUFWeightSharing, SameModelAsWithoutExtension) {
    const auto data = q4_0_bytes(2);
    const auto shared = convert_weight(data, {std::make_shared<ov::frontend::gguf::WeightSharingExtension>()});
    const auto plain = convert_weight(data);

    const auto comparator = FunctionsComparator::with_default()
                                .enable(FunctionsComparator::CONST_VALUES)
                                .enable(FunctionsComparator::NAMES)
                                .enable(FunctionsComparator::TENSOR_NAMES);
    const auto result = comparator.compare(shared, plain);
    EXPECT_TRUE(result.valid) << result.message;
}

// Without the extension every conversion extracts the weight again.
TEST(GGUFWeightSharing, NoExtensionExtractsPerConversion) {
    const auto data = q4_0_bytes(3);
    const auto first = constants(convert_weight(data));
    const auto second = constants(convert_weight(data));

    ASSERT_EQ(first.size(), second.size());
    bool any_distinct = false;
    for (size_t i = 0; i < first.size(); ++i)
        any_distinct |= first[i]->get_data_ptr() != second[i]->get_data_ptr();
    EXPECT_TRUE(any_distinct);
}

// Different weights never share a buffer.
TEST(GGUFWeightSharing, DistinctWeightsGetDistinctBuffers) {
    // Both payloads stay alive, as the extension requires: a freed one could be reallocated at the
    // same address and be taken for the first weight.
    const auto first_data = q4_0_bytes(4);
    const auto second_data = q4_0_bytes(5);
    const auto extension = std::make_shared<ov::frontend::gguf::WeightSharingExtension>();
    const auto first = constants(convert_weight(first_data, {extension}));
    const auto second = constants(convert_weight(second_data, {extension}));

    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); ++i)
        EXPECT_NE(first[i]->get_data_ptr(), second[i]->get_data_ptr()) << first[i]->get_friendly_name();
}

namespace {

ov::internal::WeightSharingCtxPtr sharing_context(const ov::frontend::gguf::WeightSharingExtension& extension) {
    const auto properties = extension.get_compile_properties();
    const auto found = properties.find(ov::internal::model_sharing_context.name());
    return found == properties.end() ? nullptr : found->second.as<ov::internal::WeightSharingCtxPtr>();
}

}  // namespace

TEST(GGUFWeightSharing, CompilePropertiesCarryTheContext) {
    const ov::frontend::gguf::WeightSharingExtension extension;
    EXPECT_TRUE(sharing_context(extension));
}

// The properties carry a copy of the context taken when they are requested: a later conversion
// does not change a context a plugin may still be reading.
TEST(GGUFWeightSharing, CompilePropertiesAreASnapshot) {
    const auto data = q4_0_bytes(6);
    const auto extension = std::make_shared<ov::frontend::gguf::WeightSharingExtension>();
    const auto before = sharing_context(*extension);
    convert_weight(data, {extension});
    const auto after = sharing_context(*extension);

    ASSERT_TRUE(before);
    ASSERT_TRUE(after);
    EXPECT_TRUE(before->m_cache_sources.empty());
    EXPECT_FALSE(after->m_cache_sources.empty());
}

// An empty weight converts without a shared buffer.
TEST(GGUFWeightSharing, EmptyWeightIsNotShared) {
    const ov::Tensor data(ov::element::u8, ov::Shape{0});
    const auto extension = std::make_shared<ov::frontend::gguf::WeightSharingExtension>();
    const auto model = SingleOpBuilder()
                           .op("GGML_OP_NONE")
                           .output("w", ov::element::f32, {0, 3})
                           .attr<ov::Tensor>("data", data)
                           .attr<std::string>("quant_type", "F32")
                           .build_with_extensions({extension});

    for (const auto& constant : constants(model))
        EXPECT_EQ(ov::weight_sharing::Extension::get_constant_source_id(*constant),
                  ov::weight_sharing::invalid_source_id)
            << constant->get_friendly_name();
    EXPECT_TRUE(sharing_context(*extension)->m_cache_sources.empty());
}

TEST(GGUFWeightSharing, RegisteringTwiceThrows) {
    ov::frontend::gguf::FrontEnd frontend;
    frontend.add_extension(std::make_shared<ov::frontend::gguf::WeightSharingExtension>());
    EXPECT_THROW(frontend.add_extension(std::make_shared<ov::frontend::gguf::WeightSharingExtension>()), ov::Exception);
}
