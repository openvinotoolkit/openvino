// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <behavior/compiled_model/import_export.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <sstream>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/codec_xor.hpp"

namespace ov {

namespace test {

namespace behavior {

inline constexpr std::string_view HostCompile_Interpreter = "HostCompile_Interpreter";

inline std::shared_ptr<ov::Model> createCustomNetModel(bool dynamicBatch = false) {
    const ov::Dimension batchDimension = dynamicBatch ? ov::Dimension(1, 10) : ov::Dimension(1);
    const ov::PartialShape inputShape{batchDimension, 16, ov::Dimension(1, 1080), ov::Dimension(10, 1920)};
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputShape);
    input->set_friendly_name("Parameter_59");

    auto make_conv_add = [](const ov::Output<ov::Node>& data,
                            const std::string& convName,
                            const std::string& addName,
                            float weightValue,
                            float biasValue) -> ov::Output<ov::Node> {
        const std::vector<float> weightValues(16 * 16, weightValue);
        const std::vector<float> biasValues(16, biasValue);

        auto weights = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{16, 16, 1, 1}, weightValues);
        auto conv = std::make_shared<ov::op::v1::Convolution>(data,
                                                              weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1},
                                                              ov::op::PadType::EXPLICIT);
        conv->set_friendly_name(convName);

        auto bias = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 16, 1, 1}, biasValues);
        auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
        add->set_friendly_name(addName);
        return add;
    };

    auto x = make_conv_add(input, "Convolution_61", "Add_63", 0.01f, 0.001f);
    x = make_conv_add(x, "Convolution_65", "Add_67", 0.011f, 0.001f);

    auto relu68 = std::make_shared<ov::op::v0::Relu>(x);
    relu68->set_friendly_name("Relu_68");
    x = relu68;

    x = make_conv_add(x, "Convolution_70", "Add_72", 0.012f, 0.001f);
    auto relu73 = std::make_shared<ov::op::v0::Relu>(x);
    relu73->set_friendly_name("Relu_73");
    x = relu73;

    x = make_conv_add(x, "Convolution_75", "Add_77", 0.013f, 0.001f);
    auto relu78 = std::make_shared<ov::op::v0::Relu>(x);
    relu78->set_friendly_name("Relu_78");
    x = relu78;

    x = make_conv_add(x, "Convolution_82", "Add_84", 0.014f, 0.001f);
    auto relu85 = std::make_shared<ov::op::v0::Relu>(x);
    relu85->set_friendly_name("Relu_85");
    x = relu85;

    x = make_conv_add(x, "Convolution_87", "Add_89", 0.015f, 0.001f);
    auto relu90 = std::make_shared<ov::op::v0::Relu>(x);
    relu90->set_friendly_name("Relu_90");
    x = relu90;

    x = make_conv_add(x, "Convolution_92", "Add_94", 0.016f, 0.001f);
    auto relu95 = std::make_shared<ov::op::v0::Relu>(x);
    relu95->set_friendly_name("Relu_95");
    x = relu95;

    auto multiplyScale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 16, 1, 1}, {0.5f});
    auto multiply97 = std::make_shared<ov::op::v1::Multiply>(x, multiplyScale);
    multiply97->set_friendly_name("Multiply_97");

    auto add98 = std::make_shared<ov::op::v1::Add>(multiply97, multiply97);
    add98->set_friendly_name("Add_98");

    x = make_conv_add(add98, "Convolution_100", "Add_102", 0.017f, 0.001f);

    auto result = std::make_shared<ov::op::v0::Result>(x);
    result->set_friendly_name("Result_104");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "CustomNet");

    // making input and output to be NHWC
    auto preProc = ov::preprocess::PrePostProcessor(model);
    preProc.input(0).tensor().set_layout("NHWC");
    preProc.input(0).model().set_layout("NCHW");
    preProc.output(0).tensor().set_layout("NHWC");
    preProc.output(0).model().set_layout("NCHW");

    model = preProc.build();

    return model;
}

using OVCompiledGraphImportExportTestNPU = OVCompiledGraphImportExportTest;

TEST_P(OVCompiledGraphImportExportTestNPU, CanImportModelWithApplicationHeaderAndTensorAPI) {
    ov::Core core;
    const std::string_view headerView("<dummy_application_header>");
    const std::string_view suffixView("<dummy_application_suffix>");
    std::stringstream sstream;

    sstream.write(headerView.data(), headerView.size());
    {
        auto model = ov::test::utils::make_conv_pool_relu();
        core.compile_model(model, target_device, configuration).export_model(sstream);
    }

    // header tests, application correctly manages offsets
    {
        auto strSO = std::make_shared<std::string>(sstream.str());
        auto tensor = ov::Tensor(ov::element::u8,
                                 ov::Shape{strSO->size() - headerView.size()},
                                 strSO->data() + headerView.size());
        auto impl = ov::get_tensor_impl(tensor);
        impl._so = strSO;
        tensor = ov::make_tensor(impl);
        sstream.seekg(headerView.size(), std::ios::beg);  // skip header

        OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(sstream, target_device, configuration));
        OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(tensor, target_device, configuration));
    }

    // suffix tests, application correctly manages ov::Tensor offsets or disables metadata checking
    {
        sstream.write(suffixView.data(), suffixView.size());
        auto strSO = std::make_shared<std::string>(sstream.str());
        auto tensor = ov::Tensor(ov::element::u8,
                                 ov::Shape{strSO->size() - headerView.size() - suffixView.size()},
                                 strSO->data() + headerView.size());
        auto impl = ov::get_tensor_impl(tensor);
        impl._so = strSO;
        tensor = ov::make_tensor(impl);
        sstream.seekg(headerView.size(), std::ios::beg);

        configuration.emplace(ov::intel_npu::disable_version_check(true));
        OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(sstream, target_device, configuration));
        configuration.erase(ov::intel_npu::disable_version_check.name());
        OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(tensor, target_device, configuration));
        sstream.seekg(headerView.size(), std::ios::beg);
        OV_EXPECT_THROW(auto compiledModel = core.import_model(sstream, target_device, configuration),
                        ov::Exception,
                        testing::HasSubstr("metadata"));  // OVNPU suffix cannot be parsed from metadata
    }
}

TEST_P(OVCompiledGraphImportExportTestNPU, CheckSizeOfRawBlobIfMultipleOfPageSize) {
    ov::Core core;
    std::stringstream sstream;

    auto rawBlobConfig = configuration;
    rawBlobConfig.emplace(ov::intel_npu::export_raw_blob(true));

    auto model = ov::test::utils::make_conv_pool_relu();
    core.compile_model(model, target_device, rawBlobConfig).export_model(sstream);

    std::size_t size = sstream.str().size();

    ASSERT_TRUE(size != 0) << "Size of the blob should be different from 0";
    ASSERT_TRUE(size % 4096 == 0) << "Size of the blob should be multiple of 4096";
}

TEST_P(OVCompiledGraphImportExportTestNPU, NonELFBlobExportThrows) {
    ov::Core core;
    std::stringstream sstream;

    auto rawBlobConfig = configuration;
    rawBlobConfig.emplace(ov::intel_npu::export_raw_blob(true));
    rawBlobConfig.emplace(ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN));

    {
        // WS
        std::stringstream modelIR, weights;
        ov::pass::Serialize serialzePass(
            modelIR,
            weights);  // serialization needed to re-read the model with WeightlessCacheAttribute set
        auto model = ov::test::utils::make_conv_pool_relu();
        serialzePass.run_on_model(model);
        const auto& weightsStr = weights.str();
        ov::Tensor weightsTensor(ov::element::u8, ov::Shape{weightsStr.size()}, weightsStr.c_str());
        model = core.read_model(modelIR.str(), weightsTensor);

        rawBlobConfig.emplace(ov::enable_weightless(true));
        OV_EXPECT_THROW(core.compile_model(model, target_device, rawBlobConfig).export_model(sstream),
                        ov::Exception,
                        testing::HasSubstr("Requested raw blob export, but the graph is not a weightful ELF one."));
        rawBlobConfig.erase(ov::enable_weightless.name());
    }

    if (ov::intel_npu::Platform::standardize(ov::test::utils::getTestPlatform()) != ov::intel_npu::Platform::NPU3720) {
        // HostCompile
        auto model = createCustomNetModel();
        rawBlobConfig.emplace(ov::intel_npu::compilation_mode(HostCompile_Interpreter));
        OV_EXPECT_THROW(core.compile_model(model, target_device, rawBlobConfig).export_model(sstream),
                        ov::Exception,
                        testing::HasSubstr("Requested raw blob export, but the graph is not a weightful ELF one."));
        rawBlobConfig.erase(ov::intel_npu::compilation_mode.name());
    }
}

TEST_P(OVCompiledGraphImportExportTestNPU, CheckSizeOfExportedModelIfMultipleOfPageSize) {
    ov::Core core;
    std::stringstream sstream;

    auto model = ov::test::utils::make_conv_pool_relu();
    core.compile_model(model, target_device, configuration).export_model(sstream);

    uint64_t size_of_blob;
    std::size_t size = sstream.str().size();

    sstream.seekg(size - std::streampos(5) /*MAGIC_BYTES*/ - sizeof(size_of_blob), std::ios::cur);
    sstream.read(reinterpret_cast<char*>(&size_of_blob), sizeof(size_of_blob));

    ASSERT_TRUE(size_of_blob != 0) << "Size of the blob shall be different from 0";
    ASSERT_TRUE(size_of_blob % 4096 == 0) << "Size of the blob shall be multiple of 4096";
}

TEST_P(OVCompiledGraphImportExportTestNPU, SameBlobAfterImportExport) {
    ov::Core core;
    std::stringstream blob_stream, test_blob_stream_1, test_blob_stream_2;

    auto model = ov::test::utils::make_conv_pool_relu();
    core.compile_model(model, target_device, configuration).export_model(blob_stream);
    configuration.insert(ov::intel_npu::defer_weights_load(true));

    core.import_model(blob_stream, target_device, configuration).export_model(test_blob_stream_1);
    ASSERT_EQ(blob_stream.str(), test_blob_stream_1.str());

    auto blob_str = blob_stream.str();
    ov::Tensor blob_tensor(ov::element::u8, ov::Shape{blob_str.size()}, blob_str.c_str());
    core.import_model(blob_tensor, target_device, configuration).export_model(test_blob_stream_2);
    ASSERT_EQ(blob_stream.str(), test_blob_stream_2.str());
}

TEST_P(OVCompiledGraphImportExportTestNPU, ImportingEncryptedBlobThrows) {
    // Encryption callbacks require L0 graph ext version >= 1.17
    NPU_SKIP_IF_GRAPH_EXT_LOWER_THAN(1, 17);

    ov::Core core;
    std::stringstream encrypted_blob_stream;

    auto model = ov::test::utils::make_conv_pool_relu();
    configuration.insert(ov::cache_encryption_callbacks(ov::EncryptionCallbacks{ov::util::codec_xor, nullptr}));
    core.compile_model(model, target_device, configuration).export_model(encrypted_blob_stream);
    auto encrypted_blob_str = encrypted_blob_stream.str();
    ov::Tensor encrypted_blob_tensor(ov::element::u8, ov::Shape{encrypted_blob_str.size()}, encrypted_blob_str.c_str());
    configuration.erase(ov::cache_encryption_callbacks.name());

    OV_EXPECT_THROW(core.import_model(encrypted_blob_stream, target_device, configuration),
                    ov::Exception,
                    ::testing::HasSubstr("Blob is encrypted, but no decryption callback was provided"));

    OV_EXPECT_THROW(core.import_model(encrypted_blob_tensor, target_device, configuration),
                    ov::Exception,
                    ::testing::HasSubstr("Blob is encrypted, but no decryption callback was provided"));

    encrypted_blob_stream.seekg(0, std::ios::beg);

    // Parsing corrupted blob on MTL will throw Access Violation 0xC0000005 SEH exceptions
    if (ov::intel_npu::Platform::standardize(ov::test::utils::getTestPlatform()) != ov::intel_npu::Platform::NPU3720) {
        configuration.insert(ov::intel_npu::import_raw_blob(true));
        OV_EXPECT_THROW(core.import_model(encrypted_blob_stream, target_device, configuration),
                        ov::Exception,
                        ::testing::HasSubstr("ZE_RESULT_ERROR_INVALID_NATIVE_BINARY"));

        OV_EXPECT_THROW(core.import_model(encrypted_blob_tensor, target_device, configuration),
                        ov::Exception,
                        ::testing::HasSubstr("ZE_RESULT_ERROR_INVALID_NATIVE_BINARY"));
    }
}

TEST_P(OVCompiledGraphImportExportTestNPU, SameUnencryptedBlobAfterDecryption) {
    ov::Core core;
    std::stringstream unencrypted_blob_stream, encrypted_blob_stream, decrypted_blob_stream;

    auto model = ov::test::utils::make_conv_pool_relu();
    core.compile_model(model, target_device, configuration).export_model(unencrypted_blob_stream);
    configuration.insert(
        ov::cache_encryption_callbacks(ov::EncryptionCallbacks{ov::util::codec_xor, ov::util::codec_xor}));
    configuration.insert(ov::intel_npu::defer_weights_load(true));
    core.import_model(unencrypted_blob_stream, target_device, configuration).export_model(encrypted_blob_stream);
    configuration.erase(ov::cache_encryption_callbacks.name());
    configuration.insert(ov::cache_encryption_callbacks(ov::EncryptionCallbacks{nullptr, ov::util::codec_xor}));

    core.import_model(encrypted_blob_stream, target_device, configuration).export_model(decrypted_blob_stream);
    ASSERT_EQ(unencrypted_blob_stream.str(), decrypted_blob_stream.str());

    decrypted_blob_stream.str(std::string());
    auto encrypted_blob_str = encrypted_blob_stream.str();
    ov::Tensor encrypted_blob_tensor(ov::element::u8, ov::Shape{encrypted_blob_str.size()}, encrypted_blob_str.c_str());
    core.import_model(encrypted_blob_tensor, target_device, configuration).export_model(decrypted_blob_stream);
    ASSERT_EQ(unencrypted_blob_stream.str(), decrypted_blob_stream.str());
}

TEST_P(OVCompiledGraphImportExportTestNPU, SameEncryptedBlobViaExportAndManualFunctionCall) {
    ov::Core core;
    std::stringstream unencrypted_blob_stream, encrypted_blob_stream;

    auto model = ov::test::utils::make_conv_pool_relu();
    // metadata is not encrypted, exclude it from blob
    configuration.insert(ov::intel_npu::import_raw_blob(true));
    configuration.insert(ov::intel_npu::export_raw_blob(true));

    configuration.insert(ov::intel_npu::defer_weights_load(true));

    core.compile_model(model, target_device, configuration).export_model(unencrypted_blob_stream);
    configuration.insert(ov::cache_encryption_callbacks(ov::EncryptionCallbacks{ov::util::codec_xor, nullptr}));
    core.import_model(unencrypted_blob_stream, target_device, configuration).export_model(encrypted_blob_stream);

    std::string manual_encrypted_blob_str = ov::util::codec_xor(unencrypted_blob_stream.str());
    std::string encrypted_blob_str = encrypted_blob_stream.str();

    ASSERT_EQ(manual_encrypted_blob_str, encrypted_blob_str);
}

TEST_P(OVCompiledGraphImportExportTestNPU, DifferentSizesOfEncryptedVsDecryptedBlobWorks) {
    // Encryption callbacks require L0 graph ext version >= 1.17
    NPU_SKIP_IF_GRAPH_EXT_LOWER_THAN(1, 17);

    ov::Core core;
    std::stringstream encrypted_blob_stream;

    std::stringstream model_xml, model_bin;
    {
        // Serialize generated model into stringstream to later populate `WeightlessCacheAttribute` runtime information
        // of constant nodes
        auto model = ov::test::utils::make_conv_pool_relu();
        ov::pass::Serialize serializer(model_xml, model_bin);
        serializer.run_on_model(model);
    }
    auto model_bin_str = model_bin.str();
    ov::Tensor model_weights(ov::element::u8, ov::Shape{model_bin_str.size()});
    std::memcpy(model_weights.data<char>(), model_bin_str.data(), model_bin_str.size());
    auto model = core.read_model(model_xml.str(), model_weights);

    configuration.insert(ov::cache_encryption_callbacks(
        ov::EncryptionCallbacks{[](const std::string& unencrypted_blob) {
                                    std::string copy_blob = unencrypted_blob;
                                    copy_blob += "<application_flag_to_mark_encryption>";
                                    return ov::util::codec_xor(copy_blob);
                                },
                                [](const std::string& encrypted_blob) {
                                    std::string decrypted_blob = ov::util::codec_xor(encrypted_blob);
                                    decrypted_blob += "<application_flag_to_mark_decryption>";
                                    return decrypted_blob;
                                }}));

    auto supported_properties = core.get_property(target_device, ov::supported_properties);
    if (std::find(supported_properties.begin(), supported_properties.end(), ov::enable_weightless.name()) !=
        supported_properties.end()) {
        configuration.insert(ov::enable_weightless(true));
    }
    OV_ASSERT_NO_THROW(core.compile_model(model, target_device, configuration).export_model(encrypted_blob_stream));

    auto encrypted_blob_str = encrypted_blob_stream.str();
    ov::Tensor encrypted_blob_tensor(ov::element::u8, ov::Shape{encrypted_blob_str.size()}, encrypted_blob_str.c_str());
    OV_ASSERT_NO_THROW(core.import_model(encrypted_blob_stream, target_device, configuration));
    OV_ASSERT_NO_THROW(core.import_model(encrypted_blob_tensor, target_device, configuration));
}

}  // namespace behavior

}  // namespace test

}  // namespace ov

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> compiledModelConfigs = {{}};

INSTANTIATE_TEST_SUITE_P(
    smoke_Behavior_NPU,
    OVCompiledGraphImportExportTestNPU,
    ::testing::Combine(::testing::Values(ov::element::f16 /* not used in internal import_export tests so far */),
                       ::testing::Values(ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(compiledModelConfigs)),
    ov::test::utils::appendPlatformTypeTestName<OVCompiledGraphImportExportTestNPU>);
