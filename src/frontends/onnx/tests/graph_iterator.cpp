// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/onnx/graph_iterator.hpp"

#include <onnx/onnx_pb.h>

#include <fstream>
#include <openvino/frontend/graph_iterator.hpp>

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

#include <openvino/openvino.hpp>

#include "../frontend/src/core/graph_iterator_proto.hpp"

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