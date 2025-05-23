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
    configuration.erase(ov::hint::compiled_blob.name());  // cleanup
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

    // header tests, stream won't work if not handled by OV caching mechanism
    {
        auto strSO = std::make_shared<std::string>(sstream.str());
        auto tensor = ov::Tensor(ov::element::u8, ov::Shape{strSO->size()}, strSO->data());
        auto impl = ov::get_tensor_impl(tensor);
        impl._so = strSO;
        tensor = ov::make_tensor(impl);
        configuration.emplace(ov::hint::compiled_blob(tensor));
        sstream.seekg(headerView.size(), std::ios::beg);  // skip header
        OV_EXPECT_THROW(
            auto compiledModel = core.import_model(sstream, target_device, configuration),
            ov::Exception,
            testing::HasSubstr("metadata"));  // OVNPU suffix can be parsed from metadata, but not correct version
        configuration.erase(ov::hint::compiled_blob.name());  // cleanup
    }

    // header tests, stream won't impact import_model if application manages ov::Tensor offset
    {
        auto strSO = std::make_shared<std::string>(sstream.str());
        auto tensor = ov::Tensor(ov::element::u8,
                                 ov::Shape{strSO->size() - headerView.size()},
                                 strSO->data() + headerView.size());
        auto impl = ov::get_tensor_impl(tensor);
        impl._so = strSO;
        tensor = ov::make_tensor(impl);
        configuration.emplace(ov::hint::compiled_blob(tensor));
        // header is no longer skipped by stream
        OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(sstream, target_device, configuration));
        configuration.erase(ov::hint::compiled_blob.name());  // cleanup
    }

    // suffix tests, stream won't impact import_model if application manages ov::Tensor size
    {
        sstream.write(suffixView.data(), suffixView.size());
        auto strSO = std::make_shared<std::string>(sstream.str());
        auto tensor = ov::Tensor(ov::element::u8,
                                 ov::Shape{strSO->size() - headerView.size() - suffixView.size()},
                                 strSO->data() + headerView.size());
        auto impl = ov::get_tensor_impl(tensor);
        impl._so = strSO;
        tensor = ov::make_tensor(impl);
        configuration.emplace(ov::hint::compiled_blob(tensor));
        OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(sstream, target_device, configuration));
        configuration.erase(ov::hint::compiled_blob.name());  // cleanup
    }
}

}  // namespace behavior

}  // namespace test

}  // namespace ov
