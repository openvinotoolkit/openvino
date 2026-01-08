// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/onnx/graph_iterator.hpp"

#include <onnx/onnx_pb.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/graph_iterator.hpp>
#include <openvino/frontend/input_model.hpp>
#include <openvino/openvino.hpp>
#include <system_error>
#include <unordered_map>

#include "../frontend/src/core/graph_iterator_proto.hpp"
#include "common_test_utils/common_utils.hpp"
#include "load_from.hpp"
#include "onnx_utils.hpp"
#include "utils.hpp"

using ::ONNX_NAMESPACE::ModelProto;
using ::ONNX_NAMESPACE::Version;

TEST_P(FrontEndLoadFromTest, testLoadUsingSimpleGraphIterator) {
    ov::frontend::FrontEnd::Ptr fe;

    class SimpleIterator : public ov::frontend::onnx::GraphIterator {
    public:
        size_t size() const override {
            return 0;
        }
        void reset() override {};
        void next() override {};
        bool is_end() const override {
            return true;
        };
        std::shared_ptr<ov::frontend::onnx::DecoderBase> get_decoder() const override {
            return nullptr;
        };

        int64_t get_opset_version(const std::string& domain) const override {
            return 1;
        }

        std::map<std::string, std::string> get_metadata() const override {
            return {};
        }

        std::string get_model_dir() const override {
            return "";
        }

        ~SimpleIterator() override {};
    };

    auto iter = std::make_shared<SimpleIterator>();

    {
        auto graph_iter = std::dynamic_pointer_cast<ov::frontend::onnx::GraphIterator>(iter);
        ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework("onnx"))
            << "Could not create the ONNX FE using a pointer GraphIterator";
        ASSERT_NE(m_frontEnd, nullptr);

        ASSERT_EQ(m_frontEnd->supported(graph_iter), true);

        ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(graph_iter)) << "Could not load the model";
        ASSERT_NE(m_inputModel, nullptr);
    }
    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel)) << "Could not convert the model to OV representation";
    ASSERT_NE(model, nullptr);

    ASSERT_EQ(model->get_ordered_ops().size(), 0);
}

TEST_P(FrontEndLoadFromTest, testLoadUsingGraphIteratorExternalStreams) {
    const std::string model_name = "external_data/external_data.onnx";
    const auto path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, model_name}).string();

    ov::frontend::FrontEnd::Ptr fe;

    auto iter = std::make_shared<ov::frontend::onnx::GraphIteratorProto>(
        ov::frontend::onnx::GraphIteratorProtoMemoryManagementMode::External_Stream);
    iter->initialize(path);
    iter->reset();

    auto graph_iter = std::dynamic_pointer_cast<ov::frontend::onnx::GraphIterator>(iter);
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework("onnx"))
        << "Could not create the ONNX FE using a pointer GraphIterator";
    ASSERT_NE(m_frontEnd, nullptr);

    ASSERT_EQ(m_frontEnd->supported(graph_iter), true);

    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(graph_iter)) << "Could not load the model";
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel)) << "Could not convert the model to OV representation";
    ASSERT_NE(model, nullptr);

    ASSERT_EQ(iter->get_mmap_cache(), nullptr);
    ASSERT_NE(iter->get_stream_cache(), nullptr);
    ASSERT_EQ(iter->get_stream_cache()->size(), 0);  // All streams must be closed after work
    ASSERT_EQ(model->get_ordered_ops().size(), 6);
}

TEST_P(FrontEndLoadFromTest, testLoadUsingGraphIteratorExternalMMAP) {
    const std::string model_name = "external_data/external_data.onnx";
    const auto path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, model_name}).string();

    ov::frontend::FrontEnd::Ptr fe;

    auto iter = std::make_shared<ov::frontend::onnx::GraphIteratorProto>(
        ov::frontend::onnx::GraphIteratorProtoMemoryManagementMode::External_MMAP);
    iter->initialize(path);
    iter->reset();

    auto graph_iter = std::dynamic_pointer_cast<ov::frontend::onnx::GraphIterator>(iter);
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework("onnx"))
        << "Could not create the ONNX FE using a pointer GraphIterator";
    ASSERT_NE(m_frontEnd, nullptr);

    ASSERT_EQ(m_frontEnd->supported(graph_iter), true);

    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(graph_iter)) << "Could not load the model";
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel)) << "Could not convert the model to OV representation";
    ASSERT_NE(model, nullptr);

    ASSERT_EQ(iter->get_stream_cache(), nullptr);
    ASSERT_NE(iter->get_mmap_cache(), nullptr);
    ASSERT_EQ(iter->get_mmap_cache()->size(), 1);  // MMAP handle must be in cache after work finished
    ASSERT_EQ(model->get_ordered_ops().size(), 6);
}

TEST_P(FrontEndLoadFromTest, tensor_place_uses_model_dir_for_external_data) {
    const std::string model_name = "external_data/external_data.onnx";
    const auto path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, model_name}).string();

    const auto original_dir = std::filesystem::path(path).parent_path();
    const auto temp_root = std::filesystem::temp_directory_path() / ov::test::utils::generateTestFilePrefix();
    ASSERT_NO_THROW(std::filesystem::create_directories(temp_root));
    struct TempDirGuard {
        std::filesystem::path dir;
        ~TempDirGuard() {
            if (dir.empty()) {
                return;
            }
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
        }
    } temp_dir_guard{temp_root};

    const auto temp_model_path = (temp_root / std::filesystem::path(path).filename()).string();

    ModelProto model_proto;
    {
        std::ifstream input(path, std::ios::binary);
        ASSERT_TRUE(input.is_open()) << "Could not open model: " << path;
        ASSERT_TRUE(model_proto.ParseFromIstream(&input)) << "Could not parse model: " << path;
    }

    std::unordered_map<std::string, std::filesystem::path> relocation_plan;
    bool updated_location = false;
    auto* initializers = model_proto.mutable_graph()->mutable_initializer();
    for (auto& tensor : *initializers) {
        for (auto& entry : *tensor.mutable_external_data()) {
            if (entry.key() == "location" && !entry.value().empty()) {
                const std::filesystem::path original_location(entry.value());
                const auto file_name = original_location.filename().string();
                ASSERT_FALSE(file_name.empty())
                    << "External data location entry has no file name component: " << entry.value();
                const auto source_path =
                    original_location.is_absolute() ? original_location : original_dir / original_location;
                relocation_plan.try_emplace(file_name, source_path);
                entry.set_value(file_name);
                updated_location = true;
            }
        }
    }
    ASSERT_TRUE(updated_location) << "External data tensor with location entry was not found";

    for (const auto& [file_name, source_path] : relocation_plan) {
        ASSERT_TRUE(std::filesystem::exists(source_path))
            << "External data file is missing in the original model directory: " << source_path;
        const auto destination_path = temp_root / file_name;
        ASSERT_NO_THROW(std::filesystem::copy_file(source_path,
                                                   destination_path,
                                                   std::filesystem::copy_options::overwrite_existing));
    }

    {
        std::ofstream output(temp_model_path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(output.is_open()) << "Could not overwrite model: " << temp_model_path;
        ASSERT_TRUE(model_proto.SerializeToOstream(&output)) << "Could not serialize model: " << temp_model_path;
    }

    auto iter = std::make_shared<ov::frontend::onnx::GraphIteratorProto>(
        ov::frontend::onnx::GraphIteratorProtoMemoryManagementMode::Internal_Stream);
    iter->initialize(temp_model_path);
    iter->reset();

    auto graph_iter = std::dynamic_pointer_cast<ov::frontend::onnx::GraphIterator>(iter);
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework("onnx"));
    ASSERT_NE(m_frontEnd, nullptr);
    ASSERT_TRUE(m_frontEnd->supported(graph_iter));

    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(graph_iter));
    ASSERT_NE(m_inputModel, nullptr);

    ASSERT_NO_THROW({
        try {
            auto model = m_frontEnd->convert(m_inputModel);
            ASSERT_NE(model, nullptr);
        } catch (const std::exception& ex) {
            std::cerr << "convert failed: " << ex.what() << std::endl;
            throw;
        }
    });
}
