// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <behavior/compiled_model/import_export.hpp>
#include <sstream>

#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/codec_xor.hpp"

namespace ov {

namespace test {

namespace behavior {

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
        OV_EXPECT_THROW(auto compiledModel = core.import_model(sstream, target_device, configuration),
                        ov::Exception,
                        testing::HasSubstr("metadata"));  // OVNPU suffix cannot be parsed from metadata
    }
}

TEST_P(OVCompiledGraphImportExportTestNPU, CheckSizeOfExportedModelIfMultipleOfPageSize) {
    ov::Core core;
    std::stringstream sstream;

    auto model = ov::test::utils::make_conv_pool_relu();
    core.compile_model(model, target_device, configuration).export_model(sstream);

    std::size_t size = sstream.str().size();

    ASSERT_TRUE(size != 0) << "Size of the exported model shall be different from 0";
    ASSERT_TRUE(size % 4096 == 0) << "Size of the exported model shall be multiple of 4096";
}

TEST_P(OVCompiledGraphImportExportTestNPU, CheckSizeOfBlobIfMultipleOfPageSize) {
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

TEST_P(OVCompiledGraphImportExportTestNPU, ImportingEncryptedBlobThrows) {
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
    core.compile_model(model, target_device, configuration).export_model(encrypted_blob_stream);
    configuration.erase(ov::cache_encryption_callbacks.name());
    configuration.insert(ov::cache_encryption_callbacks(ov::EncryptionCallbacks{nullptr, ov::util::codec_xor}));
    configuration.insert(ov::intel_npu::defer_weights_load(true));

    core.import_model(encrypted_blob_stream, target_device, configuration).export_model(decrypted_blob_stream);
    ASSERT_EQ(unencrypted_blob_stream.str(), decrypted_blob_stream.str());

    decrypted_blob_stream = {};
    auto encrypted_blob_str = encrypted_blob_stream.str();
    ov::Tensor encrypted_blob_tensor(ov::element::u8, ov::Shape{encrypted_blob_str.size()}, encrypted_blob_str.c_str());
    core.import_model(encrypted_blob_tensor, target_device, configuration).export_model(decrypted_blob_stream);
    ASSERT_EQ(unencrypted_blob_stream.str(), decrypted_blob_stream.str());
}

TEST_P(OVCompiledGraphImportExportTestNPU, SameEncryptedBlobViaExportAndManualFunctionCall) {
    ov::Core core;
    std::stringstream unencrypted_blob_stream, encrypted_blob_stream;

    auto model = ov::test::utils::make_conv_pool_relu();
    configuration.insert(ov::intel_npu::export_raw_blob(true));  // metadata is not encrypted, avoid exporting it
    core.compile_model(model, target_device, configuration).export_model(unencrypted_blob_stream);
    configuration.insert(ov::cache_encryption_callbacks(ov::EncryptionCallbacks{ov::util::codec_xor, nullptr}));
    core.compile_model(model, target_device, configuration).export_model(encrypted_blob_stream);

    std::string manual_encrypted_blob_str = ov::util::codec_xor(unencrypted_blob_stream.str());
    std::string encrypted_blob_str = encrypted_blob_stream.str();

    ASSERT_EQ(manual_encrypted_blob_str, encrypted_blob_str);
}

TEST_P(OVCompiledGraphImportExportTestNPU, DifferentSizesOfEncryptedVsDecryptedBlobWorks) {
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
