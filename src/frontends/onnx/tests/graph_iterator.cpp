// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/onnx/graph_iterator.hpp"

#include <onnx/onnx_pb.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/graph_iterator.hpp>
#include <openvino/frontend/input_model.hpp>
#include <openvino/openvino.hpp>
#include <unordered_map>
#include <vector>

#include "../frontend/src/core/decoder_proto.hpp"
#include "../frontend/src/core/graph_iterator_proto.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "load_from.hpp"
#include "onnx_utils.hpp"
#include "utils.hpp"

class SimpleIterator : public ov::frontend::onnx::GraphIterator {
public:
    mutable size_t get_model_dir_call_count = 0;
    mutable std::filesystem::path last_returned_dir;
    std::filesystem::path model_dir;

    SimpleIterator() = default;
    explicit SimpleIterator(const std::filesystem::path& dir) : model_dir(dir) {}

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

    std::filesystem::path get_model_dir() const override {
        ++get_model_dir_call_count;
        last_returned_dir = model_dir;
        return model_dir;
    }

    ~SimpleIterator() override {};
};

namespace {

class GraphIteratorProtoAccessor : public ov::frontend::onnx::GraphIteratorProto {
public:
    using ov::frontend::onnx::GraphIteratorProto::GraphIteratorProto;

    std::shared_ptr<ov::frontend::onnx::DecoderProtoTensor> get_tensor_by_name(const std::string& name) {
        GraphIteratorProto* owner = nullptr;
        return GraphIteratorProto::get_tensor(name, &owner);
    }
};

}  // namespace

TEST_P(FrontEndLoadFromTest, testLoadUsingSimpleGraphIterator) {
    ov::frontend::FrontEnd::Ptr fe;

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
    const std::filesystem::path path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, model_name});

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
    const std::filesystem::path path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, model_name});

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
    const std::filesystem::path path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, model_name});

    const auto expected_model_dir = path.parent_path();

    auto iter = std::make_shared<SimpleIterator>(expected_model_dir);

    auto graph_iter = std::dynamic_pointer_cast<ov::frontend::onnx::GraphIterator>(iter);
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework("onnx"));
    ASSERT_NE(m_frontEnd, nullptr);
    ASSERT_TRUE(m_frontEnd->supported(graph_iter));

    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(graph_iter));
    ASSERT_NE(m_inputModel, nullptr);

    try {
        auto model = m_frontEnd->convert(m_inputModel);
        ASSERT_NE(model, nullptr);
    } catch (const std::exception& ex) {
        FAIL() << "convert failed: " << ex.what();
    } catch (...) {
        FAIL() << "convert failed: reason unknown";
    }

    ASSERT_GT(iter->get_model_dir_call_count, 0) << "get_model_dir() was never called";
    ASSERT_EQ(iter->last_returned_dir, expected_model_dir)
        << "get_model_dir() returned unexpected path: " << iter->last_returned_dir
        << " (expected: " << expected_model_dir << ")";
}

TEST(FrontEndGraphIteratorTest, loads_uint16_raw_initializer_via_iterator) {
    const std::string model_name = "uint16_raw_initializer.onnx";
    const auto model_path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, model_name});

    const std::vector<uint16_t> expected_data = {100, 200, 300, 400};

    auto iterator = std::make_shared<ov::frontend::onnx::GraphIteratorProto>(
        ov::frontend::onnx::GraphIteratorProtoMemoryManagementMode::Internal_MMAP);
    iterator->initialize(model_path);
    iterator->reset();

    auto graph_iterator = std::dynamic_pointer_cast<ov::frontend::onnx::GraphIterator>(iterator);
    auto frontend = ov::frontend::FrontEndManager().load_by_framework("onnx");
    ASSERT_NE(frontend, nullptr);
    ASSERT_TRUE(frontend->supported(graph_iterator));

    auto input_model = frontend->load(graph_iterator);
    ASSERT_NE(input_model, nullptr);
    auto model = frontend->convert(input_model);
    ASSERT_NE(model, nullptr);

    ov::test::TestCase test_case(model);
    test_case.add_expected_output<uint16_t>(ov::Shape{2, 2}, expected_data);
    test_case.run();
}

TEST(FrontEndGraphIteratorTest, loads_bfloat16_raw_initializer_via_iterator) {
    const std::string model_name = "bfloat16_raw_initializer.onnx";
    const auto model_path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, model_name});

    const std::vector<ov::bfloat16> expected_data = {1.0f, 2.0f, 3.0f, 4.0f};

    auto iterator = std::make_shared<ov::frontend::onnx::GraphIteratorProto>(
        ov::frontend::onnx::GraphIteratorProtoMemoryManagementMode::Internal_MMAP);
    iterator->initialize(model_path);
    iterator->reset();

    auto graph_iterator = std::dynamic_pointer_cast<ov::frontend::onnx::GraphIterator>(iterator);
    auto frontend = ov::frontend::FrontEndManager().load_by_framework("onnx");
    ASSERT_NE(frontend, nullptr);
    ASSERT_TRUE(frontend->supported(graph_iterator));

    auto input_model = frontend->load(graph_iterator);
    ASSERT_NE(input_model, nullptr);
    auto model = frontend->convert(input_model);
    ASSERT_NE(model, nullptr);

    ov::test::TestCase test_case(model);
    test_case.add_expected_output<ov::bfloat16>(ov::Shape{2, 2}, expected_data);
    test_case.run();
}

TEST(FrontEndGraphIteratorTest, handles_optional_value_info) {
    const std::string model_name = "graph_iterator/optional_value_info.onnx";
    const auto model_path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, model_name});

    GraphIteratorProtoAccessor iterator(ov::frontend::onnx::GraphIteratorProtoMemoryManagementMode::Internal_Stream);
    ASSERT_NO_THROW(iterator.initialize(model_path));
    ASSERT_NO_THROW(iterator.reset());

    auto optional_input = iterator.get_tensor_by_name("optional_input");
    ASSERT_NE(optional_input, nullptr);
    const auto& input_info = optional_input->get_tensor_info();
    EXPECT_EQ(input_info.m_element_type, ov::element::f32);
    EXPECT_EQ(input_info.m_partial_shape, ov::PartialShape({2, 3}));

    auto optional_output = iterator.get_tensor_by_name("optional_output");
    ASSERT_NE(optional_output, nullptr);
    const auto& output_info = optional_output->get_tensor_info();
    EXPECT_EQ(output_info.m_element_type, ov::element::f32);
    EXPECT_EQ(output_info.m_partial_shape, ov::PartialShape({2, 3}));
}

TEST(FrontEndGraphIteratorTest, handles_sequence_value_info) {
    const std::string model_name = "graph_iterator/sequence_value_info.onnx";
    const auto model_path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, model_name});

    GraphIteratorProtoAccessor iterator(ov::frontend::onnx::GraphIteratorProtoMemoryManagementMode::Internal_Stream);
    ASSERT_NO_THROW(iterator.initialize(model_path));
    ASSERT_NO_THROW(iterator.reset());

    auto sequence_input = iterator.get_tensor_by_name("sequence_input");
    ASSERT_NE(sequence_input, nullptr);
    const auto& input_info = sequence_input->get_tensor_info();
    EXPECT_EQ(input_info.m_element_type, ov::element::i64);
    EXPECT_EQ(input_info.m_partial_shape, ov::PartialShape({4}));

    auto sequence_output = iterator.get_tensor_by_name("sequence_output");
    ASSERT_NE(sequence_output, nullptr);
    const auto& output_info = sequence_output->get_tensor_info();
    EXPECT_EQ(output_info.m_element_type, ov::element::i64);
    EXPECT_EQ(output_info.m_partial_shape, ov::PartialShape({4}));
}

// Test that GraphIteratorProto correctly performs topological sorting when
// nodes are not in topological order. This tests the case where a Loop node
// references an outer tensor that is produced by a node defined after the Loop.
TEST(FrontEndGraphIteratorTest, topological_sort_loop_with_unsorted_graph) {
    if (!ov::frontend::onnx::tests::is_graph_iterator_enabled()) {
        GTEST_SKIP() << "This test requires GraphIterator (ONNX_ITERATOR=1)";
    }
    const std::string model_name = "controlflow/loop_with_unsorted_graph.onnx";
    const auto model_path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, model_name});

    // Load and convert the model - this should succeed due to topological sorting
    auto frontend = ov::frontend::FrontEndManager().load_by_framework("onnx");
    ASSERT_NE(frontend, nullptr);

    auto input_model = frontend->load(model_path);
    ASSERT_NE(input_model, nullptr);

    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = frontend->convert(input_model))
        << "Model conversion failed - topological sort may not be working correctly";
    ASSERT_NE(model, nullptr);

    // Verify the model produces correct results
    // The model computes: a_final = a_init + outer_value * 3 iterations
    // where outer_value = scale * scale = 2 * 2 = 4
    // So a_final = [0, 0] + [4, 4] + [4, 4] + [4, 4] = [12, 12]
    ov::test::TestCase test_case(model);
    test_case.add_input<float>({0.f, 0.f});                                                      // a_init
    test_case.add_expected_output<float>(ov::Shape{1, 2}, {12.f, 12.f});                         // a_final
    test_case.add_expected_output<float>(ov::Shape{3, 1, 2}, {4.f, 4.f, 8.f, 8.f, 12.f, 12.f});  // a_values
    test_case.run();
}

// Helper to load and convert an ONNX model via the graph iterator path.
static std::shared_ptr<ov::Model> load_and_convert(const std::string& rel_model_path) {
    const auto model_path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, rel_model_path});
    auto frontend = ov::frontend::FrontEndManager().load_by_framework("onnx");
    auto input_model = frontend->load(model_path);
    return frontend->convert(input_model);
}

// --- Topological sort edge-case tests ---

// Linear chain in reverse order — sort must completely reorder.
TEST(FrontEndGraphIteratorTest, topological_sort_reverse_chain) {
    if (!ov::frontend::onnx::tests::is_graph_iterator_enabled()) {
        GTEST_SKIP() << "This test requires GraphIterator (ONNX_ITERATOR=1)";
    }
    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = load_and_convert("topological_sort/reverse_chain.onnx"));
    ASSERT_NE(model, nullptr);

    // Y = Neg(Abs(Relu(X))). With X=[-2, 3, -1]:
    //   Relu -> [0, 3, 0], Abs -> [0, 3, 0], Neg -> [0, -3, 0]
    ov::test::TestCase test_case(model);
    test_case.add_input<float>({-2.f, 3.f, -1.f});
    test_case.add_expected_output<float>(ov::Shape{3}, {0.f, -3.f, 0.f});
    test_case.run();
}

// Diamond pattern (unsorted) — multiple valid orderings exist.
TEST(FrontEndGraphIteratorTest, topological_sort_diamond_unsorted) {
    if (!ov::frontend::onnx::tests::is_graph_iterator_enabled()) {
        GTEST_SKIP() << "This test requires GraphIterator (ONNX_ITERATOR=1)";
    }
    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = load_and_convert("topological_sort/diamond_unsorted.onnx"));
    ASSERT_NE(model, nullptr);

    // Y = Relu(X) + Abs(X). With X=[-2, 3, -1]:
    //   R1=Relu -> [0, 3, 0], R2=Abs -> [2, 3, 1], Y = R1+R2 = [2, 6, 1]
    ov::test::TestCase test_case(model);
    test_case.add_input<float>({-2.f, 3.f, -1.f});
    test_case.add_expected_output<float>(ov::Shape{3}, {2.f, 6.f, 1.f});
    test_case.run();
}

// Unsorted node depends on initializer — initializers must be treated as known.
TEST(FrontEndGraphIteratorTest, topological_sort_initializer_dependency) {
    if (!ov::frontend::onnx::tests::is_graph_iterator_enabled()) {
        GTEST_SKIP() << "This test requires GraphIterator (ONNX_ITERATOR=1)";
    }
    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = load_and_convert("topological_sort/initializer_dep_unsorted.onnx"));
    ASSERT_NE(model, nullptr);

    // Y = Relu(X + init_val). init_val=[1,2,3], X=[-5,0,2]:
    //   T = X+init = [-4,2,5], Y = Relu(T) = [0,2,5]
    ov::test::TestCase test_case(model);
    test_case.add_input<float>({-5.f, 0.f, 2.f});
    test_case.add_expected_output<float>(ov::Shape{3}, {0.f, 2.f, 5.f});
    test_case.run();
}
