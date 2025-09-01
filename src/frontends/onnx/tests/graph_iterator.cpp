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
        size_t get_subgraph_size() const override {
            return 0;
        };

        std::shared_ptr<GraphIterator> get_subgraph(size_t idx) const override {
            return nullptr;
        };

        int64_t get_opset_version(const std::string& domain) const {
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

TEST_P(FrontEndLoadFromTest, testLoadUsingTestGraphIterator) {
    // const std::string model_name = "abs.onnx";
    // const std::string model_name = "add_abc.onnx";
    // const std::string model_name = "div.onnx";
    // const std::string model_name = "model_editor/subgraph_extraction_tests.onnx";
    // const std::string model_name = "controlflow/if_branches_with_same_inputs.onnx";
    const std::string model_name = "controlflow/if_inside_if.onnx";
    const auto path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, model_name}).string();

    ov::frontend::FrontEnd::Ptr fe;

    auto iter = std::make_shared<ov::frontend::onnx::GraphIteratorProto>(path);

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

    //ov::serialize(model, "e:/test.xml");

    ASSERT_EQ(model->get_ordered_ops().size(), 5);
}