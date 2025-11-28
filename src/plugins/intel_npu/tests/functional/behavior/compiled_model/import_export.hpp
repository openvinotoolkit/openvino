// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <behavior/compiled_model/import_export.hpp>
#include <sstream>

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
        OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(tensor, target_device, configuration));
    }

    // suffix tests, application correctly manages offsets
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

        OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(sstream, target_device, configuration));
        OV_ASSERT_NO_THROW(auto compiledModel = core.import_model(tensor, target_device, configuration));
    }
}

TEST_P(OVCompiledGraphImportExportTestNPU, CheckSizeOfExportedModelIsMultipleOfPageSize) {
    ov::Core core;
    std::stringstream sstream;

    auto model = ov::test::utils::make_conv_pool_relu();
    core.compile_model(model, target_device, configuration).export_model(sstream);

    sstream.seekg(0, std::ios::end);
    size_t sizeBlobWithMetadata = sstream.tellg();

    ASSERT_TRUE(sizeBlobWithMetadata != 0) << "Size of the exported model shall be different from 0";
    ASSERT_TRUE(sizeBlobWithMetadata % 4096 == 0) << "Size of the exported model shall be multiple of 4096";

    // same expectations for raw blob (no metadata)
    sstream = std::stringstream();

    configuration.emplace(ov::intel_npu::export_raw_blob(true));
    core.compile_model(model, target_device, configuration).export_model(sstream);
    configuration.erase(ov::intel_npu::export_raw_blob.name());

    sstream.seekg(0, std::ios::end);
    size_t sizeRawBlob = sstream.tellg();

    ASSERT_TRUE(sizeRawBlob != 0) << "Size of the blob shall be different from 0";
    ASSERT_TRUE(sizeRawBlob % 4096 == 0) << "Size of the blob shall be multiple of 4096";

    ASSERT_LE(sizeRawBlob, sizeBlobWithMetadata) << "Size of raw blob should be lesser than the one with metadata";
}

}  // namespace behavior

}  // namespace test

}  // namespace ov
