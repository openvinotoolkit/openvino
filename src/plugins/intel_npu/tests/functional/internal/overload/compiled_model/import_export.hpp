// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/import_export.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov {

namespace test {

namespace behavior {

using OVCompiledGraphImportExportTestNPU = OVCompiledGraphImportExportTest;

TEST_P(OVCompiledGraphImportExportTestNPU, CanImportModelWithEmptyIStreamAndCompiledBlobProp) {
    ov::Core core;
    std::shared_ptr<std::string> strSO;
    {
        std::stringstream sstream;
        auto model = ov::test::utils::make_conv_pool_relu();
        core.compile_model(model, target_device, configuration).export_model(sstream);
        strSO = std::make_shared<std::string>(sstream.str());
    }
    auto tensor = ov::Tensor(ov::element::u8, ov::Shape{strSO->size()}, strSO->data());
    auto impl = ov::get_tensor_impl(tensor);
    impl._so = strSO;
    tensor = ov::make_tensor(impl);
    configuration.emplace(ov::hint::compiled_blob(tensor));
    std::ifstream emptyIFileStream;
    std::fstream emptyFileStream;
    std::istringstream emptyIStringStream;
    std::stringstream emptyStringStream;
    OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(emptyIFileStream, target_device, configuration));
    OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(emptyFileStream, target_device, configuration));
    OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(emptyIStringStream, target_device, configuration));
    OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(emptyStringStream, target_device, configuration));
    configuration.erase(configuration.find(ov::hint::compiled_blob.name()));  // cleanup
}

TEST_P(OVCompiledGraphImportExportTestNPU, CanImportModelWithApplicationHeaderAndCompiledBlobProp) {
    ov::Core core;
    const std::string_view headerView("<dummy_application_header>");
    const std::string_view suffixView("<dummy_application_suffix>");
    std::stringstream sstream;

    sstream.write(headerView.data(), headerView.size());
    {
        auto model = ov::test::utils::make_conv_pool_relu();
        core.compile_model(model, target_device, configuration).export_model(sstream);
    }

    // header tests
    {
        auto strSO = std::make_shared<std::string>(sstream.str());
        auto tensor = ov::Tensor(ov::element::u8, ov::Shape{strSO->size()}, strSO->data());
        auto impl = ov::get_tensor_impl(tensor);
        impl._so = strSO;
        tensor = ov::make_tensor(impl);
        configuration.emplace(ov::hint::compiled_blob(tensor));
        sstream.seekg(headerView.size(), std::ios::beg);  // skip header
        OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(sstream, target_device, configuration));
        configuration.erase(configuration.find(ov::hint::compiled_blob.name()));
    }

    // suffix tests
    {
        sstream.write(suffixView.data(), suffixView.size());
        auto strSO = std::make_shared<std::string>(sstream.str());
        auto tensor = ov::Tensor(ov::element::u8, ov::Shape{strSO->size()}, strSO->data());
        auto impl = ov::get_tensor_impl(tensor);
        impl._so = strSO;
        tensor = ov::make_tensor(impl);
        configuration.emplace(ov::hint::compiled_blob(tensor));
        OV_EXPECT_THROW(auto compiledModel = core.import_model(sstream, target_device, configuration),
                        ov::Exception,
                        testing::HasSubstr("Blob is missing NPU metadata!"));
        configuration.emplace(ov::intel_npu::disable_version_check(true));
        OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(sstream, target_device, configuration));
    }
    // cleanup
    configuration.erase(ov::intel_npu::disable_version_check.name());
    configuration.erase(configuration.find(ov::hint::compiled_blob.name()));
}

}  // namespace behavior

}  // namespace test

}  // namespace ov
