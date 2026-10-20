// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <bitset>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>

#include "behavior/compiled_model/import_export.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "template/properties.hpp"

namespace ov::test {
namespace hsm = ov::runtime::hsm;

class TemplateHsmBlobTest : public ::testing::Test {
protected:
    static inline auto* const device = utils::DEVICE_TEMPLATE;
    static constexpr hsm::DeviceId template_device_id = 128;
    static constexpr auto blob_info_tag = hsm::SectionTag::make_device_tag(3, /*is_inline=*/true);
    static constexpr size_t blob_info_weights_embedded = 0;
    static constexpr size_t blob_info_weightless = 1;

    struct BlobInfoWire {
        uint32_t layout_version = 1;
        uint32_t flags = 0;
    };

    struct SerializedModel {
        std::string xml;
        std::shared_ptr<ov::AlignedBuffer> weights;
    };

    struct BlobSpec {
        std::string model_xml;
        bool include_model_struct = true;
        std::shared_ptr<ov::AlignedBuffer> weights;
        bool weightless = false;
        hsm::DeviceId weights_device = template_device_id;
        bool include_model_id = true;
        hsm::DeviceId model_id_device = hsm::any_device_id;
        bool reverse_order = false;
    };

    static std::shared_ptr<ov::Model> make_model() {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
        input->set_friendly_name("input");
        auto add = std::make_shared<ov::op::v1::Add>(
            input,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4}, {1.f, 2.f, 3.f, 4.f}));
        auto mul = std::make_shared<ov::op::v1::Multiply>(
            add,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4}, {0.5f, -1.f, 2.f, 3.f}));
        auto result = std::make_shared<ov::op::v0::Result>(mul);
        result->set_friendly_name("output");
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "hsm_template_model");
    }

    static std::string export_blob(ov::CompiledModel compiled) {
        std::stringstream stream;
        compiled.export_model(stream);
        return stream.str();
    }

    static ov::util::MemoryView view_of(const void* data, size_t size) {
        return {static_cast<const std::byte*>(data), size};
    }
    static ov::util::MemoryView view_of(const std::string& data) {
        return view_of(data.data(), data.size());
    }
    static ov::util::MemoryView view_of(const ov::AlignedBuffer& data) {
        return view_of(data.get_ptr(), data.size());
    }

    static SerializedModel serialize_model(const std::shared_ptr<ov::Model>& model) {
        std::stringstream xml_stream, weights_stream;
        ov::pass::Serialize(xml_stream, weights_stream).run_on_model(model);
        const auto weights_str = weights_stream.str();
        auto weights = std::make_shared<ov::AlignedBuffer>(weights_str.size());
        std::memcpy(weights->get_ptr(), weights_str.data(), weights_str.size());
        return {xml_stream.str(), weights};
    }

    static std::string build_blob(const BlobSpec& spec) {
        std::stringstream stream;
        auto writer = hsm::DeferredWriter::open(stream);
        OPENVINO_ASSERT(writer, "Cannot open HSM writer");

        const uint64_t model_id_value = 0xC0FFEE;
        std::bitset<32> blob_info_flags;
        blob_info_flags[blob_info_weights_embedded] = spec.weights != nullptr;
        blob_info_flags[blob_info_weightless] = spec.weightless;
        const BlobInfoWire blob_info{1, static_cast<uint32_t>(blob_info_flags.to_ulong())};

        std::vector<std::function<void()>> sections;
        if (spec.include_model_struct) {
            sections.push_back([&] {
                OPENVINO_ASSERT(writer->add_section(template_device_id, hsm::model_tag, view_of(spec.model_xml)));
            });
        }
        if (spec.weights) {
            sections.push_back([&] {
                OPENVINO_ASSERT(writer->add_section(spec.weights_device, hsm::weights_tag, view_of(*spec.weights)));
            });
        }
        if (spec.include_model_id) {
            sections.push_back([&] {
                OPENVINO_ASSERT(writer->add_section(spec.model_id_device,
                                                    hsm::model_id_tag,
                                                    view_of(&model_id_value, sizeof(model_id_value))));
            });
        }
        sections.push_back([&] {
            OPENVINO_ASSERT(
                writer->add_section(template_device_id, blob_info_tag, view_of(&blob_info, sizeof(blob_info))));
        });

        if (spec.reverse_order) {
            std::reverse(sections.begin(), sections.end());
        }
        for (const auto& add_section : sections) {
            add_section();
        }
        OPENVINO_ASSERT(!writer->finalize(), "HSM build failed");
        return stream.str();
    }

    static ov::Tensor infer(ov::CompiledModel& compiled) {
        const auto input =
            ov::test::utils::create_tensor<float>(ov::element::f32, ov::Shape{1, 4}, {1.f, -2.f, 3.f, -4.f});
        auto request = compiled.create_infer_request();
        request.set_input_tensor(input);
        request.infer();
        const auto output = request.get_output_tensor();
        ov::Tensor result(output.get_element_type(), output.get_shape());
        output.copy_to(result);
        return result;
    }

    static std::string read_file(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    }

    // Lazily generate one blob per suite; generateTestFilePrefix() requires an active test.
    static void generated_blob() {
        if (generated_blob_path.empty()) {
            const auto serialized = serialize_model(make_model());
            BlobSpec spec;
            spec.model_xml = serialized.xml;
            spec.weights = serialized.weights;
            const auto blob = build_blob(spec);

            generated_blob_path = std::filesystem::path(ov::test::utils::getExecutableDirectory()) /
                                  (ov::test::utils::generateTestFilePrefix() + "_hsm_generated.blob");
            std::ofstream(generated_blob_path, std::ios::binary).write(blob.data(), blob.size());
        }
    }

    static void TearDownTestSuite() {
        std::filesystem::remove(generated_blob_path);
    }

    void SetUp() override {
        core = utils::create_core(device);
        model = make_model();
        auto reference_compiled = core.compile_model(model, device);
        reference = infer(reference_compiled);
    }

    static inline std::filesystem::path generated_blob_path;

    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::Tensor reference;
};

TEST_F(TemplateHsmBlobTest, weights_path_hint_restores_model_when_weights_section_is_omitted) {
    const auto serialized = serialize_model(model);
    const auto weights_path = ov::test::utils::generateTestFilePrefix() + "_hsm_weights.bin";
    {
        std::ofstream file(weights_path, std::ios::binary);
        file.write((serialized.weights->get_ptr<char>()), static_cast<std::streamsize>(serialized.weights->size()));
    }

    BlobSpec spec;
    spec.model_xml = serialized.xml;
    const auto blob = build_blob(spec);

    std::stringstream stream(blob);
    auto imported = core.import_model(stream, device, {ov::weights_path(weights_path)});
    utils::compare(reference, infer(imported));

    std::filesystem::remove(weights_path);
}

TEST_F(TemplateHsmBlobTest, missing_mandatory_weights_section_is_rejected) {
    BlobSpec spec;
    spec.model_xml = serialize_model(model).xml;
    const auto blob = build_blob(spec);
    std::stringstream stream(blob);

    OV_EXPECT_THROW_HAS_SUBSTRING(core.import_model(stream, device), ov::Exception, "missing mandatory weights");
}

TEST_F(TemplateHsmBlobTest, missing_mandatory_model_struct_section_is_rejected) {
    const auto serialized = serialize_model(model);
    BlobSpec spec;
    spec.model_xml = serialized.xml;
    spec.include_model_struct = false;
    spec.weights = serialized.weights;
    const auto blob = build_blob(spec);
    std::stringstream stream(blob);

    OV_EXPECT_THROW_HAS_SUBSTRING(core.import_model(stream, device), ov::Exception, "missing mandatory model struct");
}

TEST_F(TemplateHsmBlobTest, model_id_section_is_optional) {
    const auto serialized = serialize_model(model);
    BlobSpec spec;
    spec.model_xml = serialized.xml;
    spec.weights = serialized.weights;
    spec.include_model_id = false;
    const auto blob = build_blob(spec);
    std::stringstream stream(blob);

    EXPECT_NO_THROW(core.import_model(stream, device));
}

TEST_F(TemplateHsmBlobTest, weights_and_model_id_sections_decode_under_any_device_id) {
    const auto serialized = serialize_model(model);
    BlobSpec spec;
    spec.model_xml = serialized.xml;
    spec.weights = serialized.weights;
    spec.weights_device = hsm::any_device_id;
    spec.model_id_device = hsm::any_device_id;
    const auto blob = build_blob(spec);

    std::stringstream stream(blob);
    auto imported = core.import_model(stream, device);

    utils::compare(reference, infer(imported));
}

TEST_F(TemplateHsmBlobTest, weights_and_model_id_sections_decode_under_template_device_id) {
    const auto serialized = serialize_model(model);
    BlobSpec spec;
    spec.model_xml = serialized.xml;
    spec.weights = serialized.weights;
    spec.weights_device = template_device_id;
    spec.model_id_device = template_device_id;
    const auto blob = build_blob(spec);

    std::stringstream stream(blob);
    auto imported = core.import_model(stream, device);

    utils::compare(reference, infer(imported));
}

TEST_F(TemplateHsmBlobTest, section_order_does_not_affect_import) {
    const auto serialized = serialize_model(model);
    BlobSpec spec;
    spec.model_xml = serialized.xml;
    spec.weights = serialized.weights;
    spec.reverse_order = true;
    const auto blob = build_blob(spec);

    std::stringstream stream(blob);
    auto imported = core.import_model(stream, device);

    utils::compare(reference, infer(imported));
}

TEST_F(TemplateHsmBlobTest, generated_blob_is_imported_from_file) {
    generated_blob();
    const auto blob = read_file(generated_blob_path);
    const auto& header = hsm::Header::view(blob.data());
    EXPECT_EQ(header.version_major, hsm::FormatVersion::major);
    EXPECT_EQ(header.version_minor, hsm::FormatVersion::minor);

    std::stringstream stream(blob);
    auto imported = core.import_model(stream, device);

    utils::compare(reference, infer(imported));
}

TEST_F(TemplateHsmBlobTest, generated_blob_is_imported_from_file_via_tensor) {
    generated_blob();
    const auto blob_tensor = ov::read_tensor_data(generated_blob_path);

    auto imported = core.import_model(blob_tensor, device);

    utils::compare(reference, infer(imported));
}
}  // namespace ov::test
