// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-corer: Apache-2.0
//

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/opsets/opset9_decl.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "openvino/opsets/opset9_decl.hpp"

namespace {

using PropertiesParams = std::tuple<std::string, std::vector<ov::AnyMap>>;

class ExportOptimalNumStreams : public ::testing::TestWithParam<PropertiesParams> {};

std::shared_ptr<ov::Model> MakeMatMulModel() {
    const ov::Shape input_shape = {1, 4096};

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(input_shape))};
    auto matmul_const = ov::test::utils::make_constant(ov::element::f32, {4096, 1024});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(params[0], matmul_const);

    auto add_const = ov::test::utils::make_constant(ov::element::f32, {1, 1024});
    auto add = ov::test::utils::make_eltwise(matmul, add_const, ov::test::utils::EltwiseTypes::ADD);
    auto softmax = std::make_shared<ov::opset9::Softmax>(add);

    ov::OutputVector results{softmax};
    return std::make_shared<ov::Model>(results, params, "MatMulModel");
}

TEST_P(ExportOptimalNumStreams, OptimalNumStreams) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    auto original_model = MakeMatMulModel();
    ov::Core core;
    const auto& [device_name, properties] = GetParam();
    auto original_properties_input = properties[0];
    auto new_properties_input = properties[1];

    auto GetProperties = [&](ov::CompiledModel& network) {
        std::vector<std::string> properties;
        properties.push_back(network.get_property(ov::hint::performance_mode.name()).as<std::string>());
        properties.push_back(network.get_property(ov::num_streams.name()).as<std::string>());
        properties.push_back(network.get_property(ov::hint::scheduling_core_type.name()).as<std::string>());
        properties.push_back(network.get_property(ov::hint::enable_hyper_threading.name()).as<std::string>());
        properties.push_back(network.get_property(ov::hint::enable_cpu_pinning.name()).as<std::string>());
        properties.push_back(network.get_property(ov::inference_num_threads.name()).as<std::string>());
        properties.push_back(network.get_property(ov::hint::num_requests.name()).as<std::string>());
        return properties;
    };

    auto original_network = core.compile_model(original_model, device_name, original_properties_input);
    auto original_properties_output = GetProperties(original_network);

    auto new_network = core.compile_model(original_model, device_name, new_properties_input);
    auto new_properties_output = GetProperties(new_network);

    std::stringstream exported_model;
    original_network.export_model(exported_model);

    // import_model with original config can create the same multi_thread setting as compile_model
    {
        std::stringstream ss(exported_model.str());
        auto imported_network = core.import_model(ss, device_name, original_properties_input);
        auto imported_properties_output = GetProperties(imported_network);

        EXPECT_EQ(original_properties_output[0], imported_properties_output[0]);
        EXPECT_EQ(original_properties_output[1], imported_properties_output[1]);
        EXPECT_EQ(original_properties_output[2], imported_properties_output[2]);
        EXPECT_EQ(original_properties_output[3], imported_properties_output[3]);
        EXPECT_EQ(original_properties_output[4], imported_properties_output[4]);
        EXPECT_EQ(original_properties_output[5], imported_properties_output[5]);
        EXPECT_EQ(original_properties_output[6], imported_properties_output[6]);
    }

    // import_model with new properties can create the same multi_thread setting as compile_model with new properties
    {
        std::stringstream ss(exported_model.str());
        auto imported_network = core.import_model(ss, device_name, new_properties_input);
        auto imported_properties_output = GetProperties(imported_network);

        EXPECT_EQ(new_properties_output[0], imported_properties_output[0]);
        EXPECT_EQ(new_properties_output[1], imported_properties_output[1]);
        EXPECT_EQ(new_properties_output[2], imported_properties_output[2]);
        EXPECT_EQ(new_properties_output[3], imported_properties_output[3]);
        EXPECT_EQ(new_properties_output[4], imported_properties_output[4]);
        EXPECT_EQ(new_properties_output[5], imported_properties_output[5]);
        EXPECT_EQ(new_properties_output[6], imported_properties_output[6]);
    }
}

const std::vector<ov::AnyMap> testing_property_for_streams = {{ov::num_streams(1)}, {ov::num_streams(2)}};

const std::vector<ov::AnyMap> testing_property_for_threads = {{ov::inference_num_threads(1)},
                                                              {ov::inference_num_threads(4)}};

const std::vector<ov::AnyMap> testing_property_for_performance_mode = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}};

const std::vector<ov::AnyMap> testing_property_for_scheduling_core_type = {
    {ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ANY_CORE)},
    {ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY)},
    {ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ECORE_ONLY)}};

const std::vector<ov::AnyMap> testing_property_for_enable_hyper_threading = {{ov::hint::enable_hyper_threading(true)},
                                                                             {ov::hint::enable_hyper_threading(false)}};

const std::vector<ov::AnyMap> testing_property_for_enable_cpu_pinning = {{ov::hint::enable_cpu_pinning(true)},
                                                                         {ov::hint::enable_cpu_pinning(false)}};

INSTANTIATE_TEST_SUITE_P(smoke_ExportImportTest,
                        ExportOptimalNumStreams,
                        ::testing::Combine(::testing::Values(std::string("CPU")),
                                           ::testing::Values(testing_property_for_streams,
                                                             testing_property_for_threads,
                                                             testing_property_for_performance_mode,
                                                             testing_property_for_scheduling_core_type,
                                                             testing_property_for_enable_hyper_threading,
                                                             testing_property_for_enable_cpu_pinning)));

TEST(CpuWeightlessImport, ValidatesConvertedWeightSizes) {
    const auto prefix = ov::test::utils::generateTestFilePrefix();
    const auto xml_path = prefix + ".xml";
    const auto bin_path = prefix + ".bin";
    struct RemoveFiles {
        std::string xml;
        std::string bin;
        ~RemoveFiles() {
            std::remove(xml.c_str());
            std::remove(bin.c_str());
        }
    } remove_files{xml_path, bin_path};

    auto parameter = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::Shape{1});
    auto weight = ov::opset9::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f});
    auto add = std::make_shared<ov::opset9::Add>(parameter, weight);
    auto result = std::make_shared<ov::opset9::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
    ov::pass::Serialize(xml_path, bin_path).run_on_model(model);

    const auto identity = [](const std::string& value) {
        return value;
    };
    ov::EncryptionCallbacks encryption{identity, identity};
    const ov::AnyMap compile_config{ov::enable_weightless(true),
                                    ov::cache_mode("OPTIMIZE_SIZE"),
                                    ov::cache_encryption_callbacks(encryption)};
    const ov::AnyMap import_config{ov::enable_weightless(true),
                                   ov::weights_path(bin_path),
                                   ov::cache_encryption_callbacks(encryption)};
    ov::Core core;
    auto compiled = core.compile_model(xml_path, "CPU", compile_config);
    std::ostringstream exported;
    compiled.export_model(exported);
    const auto blob = exported.str();

    std::istringstream control(blob);
    ASSERT_NO_THROW(core.import_model(control, "CPU", import_config));

    using Header = ov::pass::StreamSerialize::DataHeader;
    constexpr size_t prefix_size = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t);
    ASSERT_GE(blob.size(), prefix_size + sizeof(Header));
    uint64_t requirements_size = 0;
    std::memcpy(&requirements_size, blob.data() + sizeof(uint64_t) + sizeof(uint32_t), sizeof(requirements_size));
    ASSERT_LE(requirements_size, blob.size() - prefix_size - sizeof(Header));
    const auto header_pos = prefix_size + static_cast<size_t>(requirements_size);
    Header header{};
    std::memcpy(&header, blob.data() + header_pos, sizeof(header));
    ASSERT_LE(header.model_offset, blob.size() - header_pos);
    const auto model_pos = header_pos + header.model_offset;

    struct Mutation {
        const char* shape;
        const char* original_dtype;
        const char* original_size;
        const char* bin_offset;
        bool reject;
        const char* expected_error;
    };
    const auto too_many_elements = std::to_string(std::numeric_limits<size_t>::max() / 4 + 1);
    const auto same_dtype_band = std::to_string((uint64_t{1} << 59) + 1);
    const auto weights_file_size = std::filesystem::file_size(bin_path);
    ASSERT_LT(weights_file_size, static_cast<uintmax_t>(std::numeric_limits<int64_t>::max()));
    const auto beyond_weights = std::to_string(weights_file_size + 1);
    const std::array<Mutation, 7> mutations{{
        {nullptr, "FP16", nullptr, nullptr, false, nullptr},
        {too_many_elements.c_str(), nullptr, nullptr, nullptr, true, nullptr},
        {too_many_elements.c_str(), "FP16", nullptr, nullptr, true, "[CPU] Weightless cache target size overflow."},
        {nullptr, "FP16", "1", nullptr, true, "[CPU] Weightless cache source is too small."},
        {nullptr, nullptr, nullptr, beyond_weights.c_str(), true, "Incorrect weights in bin file!"},
        {nullptr, nullptr, beyond_weights.c_str(), nullptr, true, "Incorrect weights in bin file!"},
        {same_dtype_band.c_str(), nullptr, nullptr, nullptr, true, "[CPU] Weightless cache source is too small."},
    }};

    for (const auto& mutation : mutations) {
        pugi::xml_document document;
        ASSERT_EQ(document.load_buffer(blob.data() + model_pos, blob.size() - model_pos).status, pugi::status_ok);
        pugi::xml_node constant;
        pugi::xml_node weightless;
        size_t matches = 0;
        const auto find_weightless = [&](auto&& self, pugi::xml_node node) -> void {
            if (std::string(node.attribute("type").value()) == "Const") {
                for (auto attribute : node.child("rt_info").children()) {
                    if (std::string(attribute.attribute("name").value()) == "WeightlessCacheAttribute") {
                        ++matches;
                        constant = node;
                        weightless = attribute;
                    }
                }
            }
            for (auto child : node.children()) {
                self(self, child);
            }
        };
        find_weightless(find_weightless, document.document_element());
        ASSERT_EQ(matches, 1);
        ASSERT_FALSE(constant.empty());
        auto data = constant.child("data");
        ASSERT_FALSE(data.empty());
        ASSERT_FALSE(weightless.empty());
        auto shape_attr = data.attribute("shape");
        auto dtype_attr = weightless.attribute("original_dtype");
        auto size_attr = weightless.attribute("original_size");
        auto offset_attr = weightless.attribute("bin_offset");
        ASSERT_FALSE(shape_attr.empty());
        ASSERT_FALSE(dtype_attr.empty());
        ASSERT_FALSE(size_attr.empty());
        ASSERT_FALSE(offset_attr.empty());
        if (mutation.shape) {
            ASSERT_TRUE(shape_attr.set_value(mutation.shape));
        }
        if (mutation.original_dtype) {
            ASSERT_TRUE(dtype_attr.set_value(mutation.original_dtype));
        }
        if (mutation.original_size) {
            ASSERT_TRUE(size_attr.set_value(mutation.original_size));
        }
        if (mutation.bin_offset) {
            ASSERT_TRUE(offset_attr.set_value(mutation.bin_offset));
        }
        std::ostringstream changed_xml;
        document.save(changed_xml);
        auto changed_header = header;
        changed_header.model_size = changed_xml.str().size();
        auto changed_blob = blob.substr(0, model_pos) + changed_xml.str();
        std::memcpy(changed_blob.data() + header_pos, &changed_header, sizeof(changed_header));
        std::istringstream changed_stream(changed_blob);
        if (mutation.expected_error) {
            try {
                core.import_model(changed_stream, "CPU", import_config);
                FAIL() << "Expected rejection of invalid weightless metadata";
            } catch (const ov::Exception& error) {
                EXPECT_NE(std::string(error.what()).find(mutation.expected_error), std::string::npos);
            }
        } else if (mutation.reject) {
            EXPECT_THROW(core.import_model(changed_stream, "CPU", import_config), ov::Exception);
        } else {
            EXPECT_NO_THROW(core.import_model(changed_stream, "CPU", import_config));
        }
    }
}

}  // namespace
