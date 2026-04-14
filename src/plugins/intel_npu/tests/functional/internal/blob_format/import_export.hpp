// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "compiler_schedules_sections.hpp"

#include "common/functions.hpp"

namespace ov::test::behavior {
    using namespace ::intel_npu;

    TEST(BlobFormat, MultipleImportExportSameModel) {
        ov::Core core;
        std::shared_ptr<ov::Model> model = buildSingleLayerSoftMaxNetwork();

        CompiledModel compiled_model;
        OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, "NPU", {}));

        std::stringstream stream;
        compiled_model.export_model(stream);

        auto blobSize = BlobReader::get_npu_region_size(stream);

        const std::string& str = stream.str();
        ov::Tensor tensor(ov::element::u8, ov::Shape{str.size()});
        std::memcpy(tensor.data(), str.data(), str.size());

        OV_ASSERT_NO_THROW(compiled_model = core.import_model(tensor, "NPU", {{ov::hint::compiled_blob.name(), true}}));

        std::stringstream stream2;
        compiled_model.export_model(stream2);

        auto blobSize2 = BlobReader::get_npu_region_size(stream2);
        EXPECT_EQ(blobSize, blobSize2);
    }

    TEST(BlobFormat, RepeatedImportExportSameModel) {
        ov::Core core;
        std::shared_ptr<ov::Model> model = buildSingleLayerSoftMaxNetwork();

        CompiledModel compiled_model;
        OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, "NPU", {}));

        std::stringstream initial_stream;
        compiled_model.export_model(initial_stream);
        const std::string initial_str = initial_stream.str();
        auto initialBlobSize = BlobReader::get_npu_region_size(initial_stream);

        std::string current_str = initial_str;
        for (int round = 1; round <= 4; ++round) {
            ov::Tensor tensor(ov::element::u8, ov::Shape{current_str.size()});
            std::memcpy(tensor.data(), current_str.data(), current_str.size());

            OV_ASSERT_NO_THROW(compiled_model = core.import_model(tensor, "NPU", {{ov::hint::compiled_blob.name(), true}}));

            std::stringstream out_stream;
            compiled_model.export_model(out_stream);
            current_str = out_stream.str();

            auto blobSize = BlobReader::get_npu_region_size(out_stream);
            EXPECT_EQ(initialBlobSize, blobSize) << "Blob size mismatch at round " << round;
        }
    }

}
