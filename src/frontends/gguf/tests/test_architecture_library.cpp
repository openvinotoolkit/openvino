// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <filesystem>
#include <numeric>

#include "common_test_utils/file_utils.hpp"
#include "gguf_writer.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/gguf/frontend.hpp"
#include "projector.hpp"

// Link the actual frontend library to exercise the SDK and module-loading boundary.
TEST(GGUFArchitectureLibrary, ExternalModuleMatchesDirectRegistrationAcrossTokenCounts) {
    const auto path =
        std::filesystem::temp_directory_path() / (ov::test::utils::generateTestFilePrefix() + "_projector.gguf");
    struct RemoveFile {
        std::filesystem::path path;
        ~RemoveFile() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup{path};
    ov_gguf_test::GgufWriter writer;
    writer.kv_str("general.architecture", "example-projector");
    writer.tensor("projection.weight", {2, 3}, {1, 2, 3, 4, 5, 6});
    ASSERT_TRUE(writer.write(path.string()));

    std::shared_ptr<ov::Model> from_library;
    {
        ov::frontend::gguf::FrontEnd frontend;
        const auto library = std::filesystem::path(ov::test::utils::getExecutableDirectory()) / ARCH_EXTENSION_LIBRARY;
        static_cast<ov::frontend::FrontEnd&>(frontend).add_extension(library.string());
        from_library = frontend.convert(frontend.load(path.string()));
    }
    ov::frontend::gguf::FrontEnd direct;
    direct.add_extension(
        std::make_shared<ov::frontend::gguf::ArchitectureExtension>(example::projector_architecture()));
    auto from_definition = direct.convert(direct.load(path.string()));
    for (size_t tokens : {1u, 3u, 7u}) {
        ov::Tensor data(ov::element::f32, {1, 1, tokens, 2});
        std::iota(data.data<float>(), data.data<float>() + data.get_size(), 1.0f);
        ov::TensorVector a{ov::Tensor(ov::element::f32, {1, 1, tokens, 3})};
        ov::TensorVector b{ov::Tensor(ov::element::f32, {1, 1, tokens, 3})};
        ASSERT_TRUE(from_library->evaluate(a, {data}));
        ASSERT_TRUE(from_definition->evaluate(b, {data}));
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t channel = 0; channel < 3; ++channel) {
                const auto expected = float((2 * t + 1) * (2 * channel + 1) + (2 * t + 2) * (2 * channel + 2));
                EXPECT_FLOAT_EQ(a[0].data<float>()[t * 3 + channel], expected);
                EXPECT_FLOAT_EQ(b[0].data<float>()[t * 3 + channel], expected);
            }
        }
    }
}
