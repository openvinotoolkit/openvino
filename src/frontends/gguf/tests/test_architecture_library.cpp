// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <filesystem>
#include <numeric>

#include "../src/model_adapter_attribute.hpp"
#include "common_test_utils/file_utils.hpp"
#include "gguf_writer.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "projector.hpp"

// Link the actual frontend library to exercise the builder API and module-loading boundary.
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

TEST(GGUFArchitectureLibrary, ProjectorAdapterRetainsItsLibraryUntilExecutionCompletes) {
    const auto path =
        std::filesystem::temp_directory_path() / (ov::test::utils::generateTestFilePrefix() + "_adapter.gguf");
    struct RemoveFile {
        std::filesystem::path path;
        ~RemoveFile() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup{path};
    ov_gguf_test::GgufWriter writer;
    writer.kv_str("general.architecture", "clip");
    writer.kv_bool("clip.has_vision_encoder", true);
    writer.kv_str("clip.vision.projector_type", "example-adapter");
    writer.tensor("projection.weight", {2, 3}, {1, 2, 3, 4, 5, 6});
    ASSERT_TRUE(writer.write(path.string()));
    std::shared_ptr<ov::frontend::gguf::ModelAdapter> holder;
    std::shared_ptr<ov::Model> model;
    {
        ov::frontend::gguf::FrontEnd frontend;
        const auto library = std::filesystem::path(ov::test::utils::getExecutableDirectory()) / ARCH_EXTENSION_LIBRARY;
        static_cast<ov::frontend::FrontEnd&>(frontend).add_extension(library.string());
        model = frontend.convert(frontend.load(path.string()));
    }
    holder = ov::frontend::gguf::get_model_adapter(model->get_results().front());
    ASSERT_NE(holder, nullptr);
    ASSERT_FALSE(holder->lifetime.empty());
    std::weak_ptr<ov::Extension> owner = holder->lifetime.front();
    EXPECT_FALSE(owner.expired());
    using Adapter = ov::frontend::gguf::pass::AdaptMmprojToGenAI;
    Adapter adapter(Adapter::Modality::VISION, Adapter::Layout::VISION_ENCODERS);
    ASSERT_TRUE(adapter.run_on_model(model));
    EXPECT_EQ(adapter.get_vision_models().at("custom_vision"), model);
    EXPECT_EQ(ov::frontend::gguf::get_model_adapter(model->get_results().front()), nullptr);
    for (const size_t tokens : {1u, 3u, 7u}) {
        ov::Tensor input(ov::element::f32, {1, 1, tokens, 2});
        std::iota(input.data<float>(), input.data<float>() + input.get_size(), 1.f);
        ov::TensorVector outputs{ov::Tensor(ov::element::f32, {1, tokens, 3})};
        ASSERT_TRUE(model->evaluate(outputs, {input}));
        for (size_t t = 0; t < tokens; ++t)
            for (size_t c = 0; c < 3; ++c)
                EXPECT_FLOAT_EQ(outputs[0].data<float>()[t * 3 + c],
                                float((2 * t + 1) * (2 * c + 1) + (2 * t + 2) * (2 * c + 2) + 7));
    }
    adapter = Adapter(Adapter::Modality::VISION, Adapter::Layout::VISION_ENCODERS);
    model.reset();
    holder.reset();
    EXPECT_TRUE(owner.expired());
}
