// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>

#include "behavior/compiled_model/import_export.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"

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
        // OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(tensor, target_device, configuration));
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
        // OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(tensor, target_device, configuration));
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

}  // namespace behavior

}  // namespace test

}  // namespace ov
