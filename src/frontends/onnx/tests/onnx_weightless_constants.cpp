// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "../frontend/src/core/graph_iterator_proto.hpp"
#include "../frontend/src/utils/weightless_constants.hpp"
#include "common_test_utils/file_utils.hpp"
#include "onnx_utils.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;
using namespace ov::frontend::onnx;
namespace weightless = ov::frontend::onnx::weightless;

namespace {

std::string model_path(const std::string& name) {
    return test::utils::getModelFromTestModelZoo(std::string(TEST_ONNX_MODELS_DIRNAME) + name);
}

// Friendly name -> WeightlessCacheAttribute of the model's Constants (nullopt when absent).
std::map<std::string, std::optional<WeightlessCacheAttribute>> get_weightless_attributes(
    const std::shared_ptr<Model>& model) {
    std::map<std::string, std::optional<WeightlessCacheAttribute>> result;
    for (const auto& node : model->get_ordered_ops()) {
        const auto constant = as_type_ptr<op::v0::Constant>(node);
        if (!constant) {
            continue;
        }
        const auto& rt_info = constant->get_rt_info();
        const auto it = rt_info.find(WeightlessCacheAttribute::get_type_info_static());
        result[constant->get_friendly_name()] =
            it == rt_info.end() ? std::nullopt : std::make_optional(it->second.as<WeightlessCacheAttribute>());
    }
    return result;
}

void expect_no_initializer_ids(const std::shared_ptr<Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        EXPECT_EQ(node->get_rt_info().count(weightless::OnnxInitializerId::get_type_info_static()), 0)
            << "OnnxInitializerId must not be visible after conversion: " << node->get_friendly_name();
    }
}

std::shared_ptr<op::v0::Constant> get_constant(const std::shared_ptr<Model>& model, const std::string& name) {
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto constant = as_type_ptr<op::v0::Constant>(node);
            constant && constant->get_friendly_name() == name) {
            return constant;
        }
    }
    return nullptr;
}

void expect_synthetic_id(const std::shared_ptr<Model>& model, const std::string& name) {
    const auto constant = get_constant(model, name);
    ASSERT_NE(constant, nullptr) << name;
    const auto attributes = get_weightless_attributes(model);
    const auto& wca = attributes.at(name);
    ASSERT_TRUE(wca.has_value()) << "Constant '" << name << "' is expected to have a synthetic ID";
    EXPECT_EQ(wca->bin_offset,
              weightless::make_synthetic_id(name, constant->get_byte_size(), constant->get_element_type()));
    EXPECT_EQ(wca->original_size, constant->get_byte_size());
    EXPECT_EQ(wca->original_dtype, constant->get_element_type());
}

// A GraphIterator over an ONNX file that also reports configured provenance, like OVEP's ORT delegate.
class ProvenanceIterator : public GraphIteratorProto, public TensorOriginProvider {
public:
    ProvenanceIterator() : GraphIteratorProto(GraphIteratorProtoMemoryManagementMode::Internal_MMAP) {}

    SourceModelLocation get_source_model_location(const std::string& tensor_name) const override {
        const auto it = locations.find(tensor_name);
        return it == locations.end() ? SourceModelLocation::UNKNOWN : it->second;
    }

    std::optional<SourceModelExternalData> get_source_model_external_data(
        const std::string& tensor_name) const override {
        const auto it = declared.find(tensor_name);
        return it == declared.end() ? std::nullopt : std::make_optional(it->second);
    }

    std::map<std::string, SourceModelLocation> locations;
    std::map<std::string, SourceModelExternalData> declared;
};

std::shared_ptr<Model> convert_with_iterator(const std::shared_ptr<ProvenanceIterator>& iterator,
                                             const std::string& name,
                                             WeightlessConstants mode) {
    iterator->initialize(model_path(name));
    iterator->reset();
    frontend::FrontEndManager manager;
    const auto frontend = manager.load_by_framework("onnx");
    const auto input_model = frontend->load(std::static_pointer_cast<GraphIterator>(iterator),
                                            true,
                                            false,
                                            AnyMap{weightless_constants(mode)});
    return frontend->convert(input_model);
}

}  // namespace

// Scheme 1 is a format contract: these IDs are stored in compiled blobs, so they must never change.
TEST(OnnxWeightlessConstants, synthetic_id_known_answers) {
    EXPECT_EQ(weightless::make_synthetic_id("A", 16, element::f32), 0x991983ce14e4839aull);
    EXPECT_EQ(weightless::make_synthetic_id("B", 16, element::f32), 0x938199ed2a00f5daull);
    EXPECT_EQ(weightless::make_synthetic_id("model.layers.0.mlp.down_proj.weight", 4096, element::f16),
              0x989647b43401cbe3ull);
    EXPECT_EQ(weightless::make_synthetic_id("", 0, element::u8), 0x9340d812367d73bbull);
}

TEST(OnnxWeightlessConstants, synthetic_id_layout) {
    for (const auto& name : {"A", "B", "w", "model.layers.0.mlp.down_proj.weight"}) {
        const auto id = weightless::make_synthetic_id(name, 64, element::f32);
        EXPECT_TRUE(weightless::is_synthetic_id(id)) << name;
        EXPECT_EQ(weightless::get_id_scheme(id), weightless::kIdScheme) << name;
    }
    // Every input field changes the ID.
    const auto base = weightless::make_synthetic_id("w", 64, element::f32);
    EXPECT_NE(base, weightless::make_synthetic_id("w2", 64, element::f32));
    EXPECT_NE(base, weightless::make_synthetic_id("w", 32, element::f32));
    EXPECT_NE(base, weightless::make_synthetic_id("w", 64, element::i32));
}

TEST(OnnxWeightlessConstants, default_mode_is_unchanged) {
    Core core;
    const auto model = core.read_model(model_path("external_data/external_data.onnx"));
    const auto attributes = get_weightless_attributes(model);
    ASSERT_TRUE(attributes.at("A").has_value());
    EXPECT_EQ(attributes.at("A")->bin_offset, 0);
    EXPECT_FALSE(attributes.at("B").has_value());
    EXPECT_FALSE(model->has_rt_info(weightless::kRtInfoWeightlessConstants));
    expect_no_initializer_ids(model);
}

TEST(OnnxWeightlessConstants, all_mode_inline_model) {
    Core core;
    const auto model =
        core.read_model(model_path("add_abc_initializers.onnx"), {}, {weightless_constants(WeightlessConstants::ALL)});
    // "A" is an initializer and "B" the output of a Constant op: both are stored inside the model.
    expect_synthetic_id(model, "A");
    expect_synthetic_id(model, "B");
    EXPECT_EQ(model->get_rt_info<WeightlessConstants>(weightless::kRtInfoWeightlessConstants),
              WeightlessConstants::ALL);
    EXPECT_EQ(model->get_rt_info<uint64_t>(weightless::kRtInfoIdScheme), weightless::kIdScheme);
    expect_no_initializer_ids(model);
}

TEST(OnnxWeightlessConstants, all_mode_mixed_model) {
    Core core;
    const auto model = core.read_model(model_path("external_data/external_data.onnx"),
                                       {},
                                       {weightless_constants(WeightlessConstants::ALL)});
    const auto attributes = get_weightless_attributes(model);
    // The external initializer keeps its real offset; the Constant op gets a synthetic ID.
    ASSERT_TRUE(attributes.at("A").has_value());
    EXPECT_EQ(attributes.at("A")->bin_offset, 0);
    EXPECT_FALSE(weightless::is_synthetic_id(attributes.at("A")->bin_offset));
    expect_synthetic_id(model, "B");
    expect_no_initializer_ids(model);
}

TEST(OnnxWeightlessConstants, all_mode_frontend_load_option) {
    frontend::FrontEndManager manager;
    const auto frontend = manager.load_by_framework("onnx");
    // The option can follow the positional parameters.
    const auto input_model = frontend->load(model_path("add_abc_initializers.onnx"),
                                            true,
                                            AnyMap{weightless_constants(WeightlessConstants::ALL)});
    const auto model = frontend->convert(input_model);
    expect_synthetic_id(model, "A");
    expect_synthetic_id(model, "B");
}

TEST(OnnxWeightlessConstants, all_mode_same_ids_from_graph_iterator) {
    Core core;
    const auto from_file =
        core.read_model(model_path("add_abc_initializers.onnx"), {}, {weightless_constants(WeightlessConstants::ALL)});
    const auto from_iterator = convert_with_iterator(std::make_shared<ProvenanceIterator>(),
                                                     "add_abc_initializers.onnx",
                                                     WeightlessConstants::ALL);
    const auto expected = get_weightless_attributes(from_file);
    const auto actual = get_weightless_attributes(from_iterator);
    for (const auto& name : {"A", "B"}) {
        ASSERT_TRUE(expected.at(name).has_value()) << name;
        ASSERT_TRUE(actual.at(name).has_value()) << name;
        EXPECT_EQ(expected.at(name)->bin_offset, actual.at(name)->bin_offset) << name;
    }
    expect_no_initializer_ids(from_iterator);
}

TEST(OnnxWeightlessConstants, provider_none_is_embedded) {
    auto iterator = std::make_shared<ProvenanceIterator>();
    iterator->locations = {{"A", SourceModelLocation::NONE}};
    const auto model = convert_with_iterator(iterator, "external_data/external_data.onnx", WeightlessConstants::ALL);
    // "A" is read from a file, but the source model doesn't provide it: its file offset is removed.
    EXPECT_FALSE(get_weightless_attributes(model).at("A").has_value());
    EXPECT_EQ(model->get_rt_info<uint64_t>("weightless_stats", "unbacked_count"), 1);
    // "B" is a Constant op of the iterated graph, which the source model provides the same way.
    expect_synthetic_id(model, "B");
}

TEST(OnnxWeightlessConstants, provider_external_uses_declared_offset) {
    auto iterator = std::make_shared<ProvenanceIterator>();
    iterator->locations = {{"A", SourceModelLocation::EXTERNAL}};
    iterator->declared = {{"A", SourceModelExternalData{"tensors_data/tensor.data", 4096, 0}}};
    const auto model =
        convert_with_iterator(iterator, "external_data/external_data.onnx", WeightlessConstants::EXTERNAL);
    const auto attributes = get_weightless_attributes(model);
    ASSERT_TRUE(attributes.at("A").has_value());
    EXPECT_EQ(attributes.at("A")->bin_offset, 4096);
    // With a provider the index-based fallback is never used.
    EXPECT_FALSE(attributes.at("B").has_value());
    EXPECT_EQ(model->get_rt_info<WeightlessConstants>(weightless::kRtInfoWeightlessConstants),
              WeightlessConstants::EXTERNAL);
}

TEST(OnnxWeightlessConstants, provider_external_without_declared_location_is_embedded) {
    auto iterator = std::make_shared<ProvenanceIterator>();
    iterator->locations = {{"A", SourceModelLocation::EXTERNAL}};
    const auto model = convert_with_iterator(iterator, "external_data/external_data.onnx", WeightlessConstants::ALL);
    EXPECT_FALSE(get_weightless_attributes(model).at("A").has_value());
}

TEST(OnnxWeightlessConstants, provider_external_length_mismatch_is_embedded) {
    auto iterator = std::make_shared<ProvenanceIterator>();
    iterator->locations = {{"A", SourceModelLocation::EXTERNAL}};
    iterator->declared = {{"A", SourceModelExternalData{"tensors_data/tensor.data", 0, 8}}};
    const auto model = convert_with_iterator(iterator, "external_data/external_data.onnx", WeightlessConstants::ALL);
    EXPECT_FALSE(get_weightless_attributes(model).at("A").has_value());
}

TEST(OnnxWeightlessConstants, provider_internal_from_file_is_embedded) {
    auto iterator = std::make_shared<ProvenanceIterator>();
    iterator->locations = {{"A", SourceModelLocation::INTERNAL}};
    const auto model = convert_with_iterator(iterator, "external_data/external_data.onnx", WeightlessConstants::ALL);
    EXPECT_FALSE(get_weightless_attributes(model).at("A").has_value());
}

TEST(OnnxWeightlessConstants, provider_internal_gets_synthetic_id) {
    auto iterator = std::make_shared<ProvenanceIterator>();
    iterator->locations = {{"A", SourceModelLocation::INTERNAL}};
    const auto model = convert_with_iterator(iterator, "add_abc_initializers.onnx", WeightlessConstants::ALL);
    expect_synthetic_id(model, "A");
}

TEST(OnnxWeightlessConstants, external_mode_with_provider_skips_index_fallback) {
    // Without external data the index-based fallback would give every constant an ID. With a provider it must not.
    auto iterator = std::make_shared<ProvenanceIterator>();
    iterator->locations = {{"A", SourceModelLocation::NONE}};
    const auto model = convert_with_iterator(iterator, "add_abc_initializers.onnx", WeightlessConstants::EXTERNAL);
    for (const auto& [name, wca] : get_weightless_attributes(model)) {
        EXPECT_FALSE(wca.has_value()) << "'" << name << "' must not get an index-based ID";
    }
    EXPECT_TRUE(model->has_rt_info(weightless::kRtInfoWeightlessConstants));
    expect_no_initializer_ids(model);
}
