// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "load_from.hpp"

#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/test_case.hpp"
#include "core/graph_iterator_proto.hpp"
#include "onnx_utils.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/frontend/extension/progress_reporter.hpp"
#include "openvino/frontend/onnx/extension/conversion.hpp"
#include "openvino/frontend/onnx/frontend.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

using namespace ov::frontend;

using ONNXLoadTest = FrontEndLoadFromTest;
using testing::Contains;
using testing::ElementsAre;
using testing::Property;
using testing::UnorderedElementsAre;

static LoadFromFEParam getTestData() {
    LoadFromFEParam res;
    res.m_frontEndName = ONNX_FE;
    res.m_modelsPath = std::string(TEST_ONNX_MODELS_DIRNAME);
    res.m_file = "external_data/external_data.onnx";
    res.m_stream = "add_abc.onnx";
    return res;
}

TEST_P(FrontEndLoadFromTest, testLoadFromStreamAndPassPath) {
    const auto path = ov::util::path_join(
        {ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "external_data/external_data.onnx"});
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    ASSERT_TRUE(ifs.is_open()) << "Could not open an ifstream for the model path: " << path;
    std::istream* is = &ifs;
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    OV_ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_model(is)) << "Could not create the ONNX FE using the istream object";
    ASSERT_NE(m_frontEnd, nullptr);

    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(is, path)) << "Could not load the model";
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ov::Model> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel)) << "Could not convert the model to OV representation";
    ASSERT_NE(function, nullptr);
}

TEST_P(FrontEndLoadFromTest, load_model_not_exists_at_path) {
    const auto model_name = "not_existing_model";
    auto error_msg = std::string("Could not open the file: ");
    auto model_file_path = FrontEndTestUtils::make_model_path(model_name);
    error_msg += '"' + model_file_path + '"';

    auto fem = ov::frontend::FrontEndManager();
    auto fe = fem.load_by_framework("onnx");

    OV_EXPECT_THROW(fe->supported({model_file_path}), ov::Exception, testing::HasSubstr(error_msg));
    OV_EXPECT_THROW(fe->load(model_file_path), ov::Exception, testing::HasSubstr(error_msg));
}

TEST_P(FrontEndLoadFromTest, load_model_and_apply_ppp) {
    auto model_file_path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, m_param.m_stream});

    m_frontEnd = m_fem.load_by_model(model_file_path);
    const auto fe_model = m_frontEnd->load(model_file_path);
    auto model = m_frontEnd->convert(fe_model);

    EXPECT_THAT(model->inputs(),
                ElementsAre(Property("Input 0", &ov::Output<ov::Node>::get_names, UnorderedElementsAre("A")),
                            Property("Input 1", &ov::Output<ov::Node>::get_names, UnorderedElementsAre("B")),
                            Property("Input 2", &ov::Output<ov::Node>::get_names, UnorderedElementsAre("C"))));
    EXPECT_THAT(model->output(0).get_names(), UnorderedElementsAre("Y"));

    auto p = ov::preprocess::PrePostProcessor(model);
    p.output(0).tensor().set_element_type(ov::element::f16);
    model = p.build();

    EXPECT_THAT(model->inputs(),
                ElementsAre(Property("Input 0", &ov::Output<ov::Node>::get_names, UnorderedElementsAre("A")),
                            Property("Input 1", &ov::Output<ov::Node>::get_names, UnorderedElementsAre("B")),
                            Property("Input 2", &ov::Output<ov::Node>::get_names, UnorderedElementsAre("C"))));
    EXPECT_THAT(model->output(0).get_names(), UnorderedElementsAre("Y"));
}

INSTANTIATE_TEST_SUITE_P(ONNXLoadTest,
                         FrontEndLoadFromTest,
                         ::testing::Values(getTestData()),
                         FrontEndLoadFromTest::getTestCaseName);

// !!! Experimental feature, it may be changed or removed in the future !!!
using ::ONNX_NAMESPACE::ModelProto;
using ::ONNX_NAMESPACE::Version;

TEST_P(FrontEndLoadFromTest, testLoadFromModelProtoUint64) {
    const auto path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "abs.onnx"});
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    ASSERT_TRUE(ifs.is_open()) << "Could not open an ifstream for the model path: " << path;
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;

    {
        auto model_proto = std::make_shared<ModelProto>();
        ASSERT_TRUE(model_proto->ParseFromIstream(&ifs)) << "Could not parse ModelProto from file: " << path;

        uint64_t model_proto_ptr = reinterpret_cast<uint64_t>(model_proto.get());

        ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_model(model_proto_ptr))
            << "Could not create the ONNX FE using a pointer on ModelProto object as uint64_t";
        ASSERT_NE(m_frontEnd, nullptr);
        ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(model_proto_ptr)) << "Could not load the model";
        ASSERT_NE(m_inputModel, nullptr);
    }

    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel)) << "Could not convert the model to OV representation";
    ASSERT_NE(model, nullptr);

    ASSERT_TRUE(model->get_ordered_ops().size() > 0);
}

TEST_P(FrontEndLoadFromTest, testLoadFromModelProtoUint64_Negative) {
    const auto path =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "abs.onnx"});
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    ASSERT_TRUE(ifs.is_open()) << "Could not open an ifstream for the model path: " << path;
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;

    auto model_proto = std::make_shared<ModelProto>();
    ASSERT_TRUE(model_proto->ParseFromIstream(&ifs)) << "Could not parse ModelProto from file: " << path;

    uint64_t model_proto_ptr = reinterpret_cast<uint64_t>(model_proto.get());

    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_model(model_proto_ptr))
        << "Could not create the ONNX FE using a pointer on ModelProto object as uint64_t";
    ASSERT_NE(m_frontEnd, nullptr);
    // Should say unsupported if an address is 0
    ASSERT_FALSE(m_frontEnd->supported(static_cast<uint64_t>(0)));
    // Should throw an ov::Exception if address is 0
    OV_EXPECT_THROW(m_inputModel = m_frontEnd->load(static_cast<uint64_t>(0)),
                    ov::Exception,
                    testing::HasSubstr("Wrong address"));

    model_proto->set_ir_version(Version::IR_VERSION + 1);
    // Should say unsupported if ModelProto has IR_VERSION higher than supported
    ASSERT_FALSE(m_frontEnd->supported(model_proto_ptr));
    // Should throw an ov::Exception if address is 0
    OV_EXPECT_THROW(m_inputModel = m_frontEnd->load(model_proto_ptr),
                    ov::Exception,
                    testing::HasSubstr("unsupported IR version"));
}
// !!! End of Experimental feature !!!

TEST(FrontEndInputModel, InitializersAreNotModelInputs) {
    ov::frontend::FrontEnd::Ptr fe;
    auto input_model = ov::frontend::onnx::tests::load_model("regression/initializer_shares_input_name.onnx", &fe);
    ASSERT_NE(input_model, nullptr);

    auto inputs = input_model->get_inputs();
    ASSERT_EQ(inputs.size(), 1);

    std::vector<std::string> input_names;
    input_names.reserve(inputs.size());
    for (const auto& place : inputs) {
        ASSERT_FALSE(place->get_names().empty());
        input_names.push_back(place->get_names().front());
    }

    EXPECT_THAT(input_names, ElementsAre("data"));
}

namespace {

// Exercise both the default and explicit iterator setting, as well as the legacy fallback.
class ONNXInMemoryLoadTest : public testing::TestWithParam<std::tuple<const char*, bool>> {
protected:
    std::optional<std::string> previous_iterator_setting;

    static int set_iterator_setting(const char* value) {
#ifdef _WIN32
        return _putenv_s("ONNX_ITERATOR", value ? value : "");
#else
        return value ? setenv("ONNX_ITERATOR", value, 1) : unsetenv("ONNX_ITERATOR");
#endif
    }

    void SetUp() override {
        if (const auto value = std::getenv("ONNX_ITERATOR")) {
            previous_iterator_setting = value;
        }
        ASSERT_EQ(set_iterator_setting(std::get<0>(GetParam())), 0);
    }

    void TearDown() override {
        EXPECT_EQ(set_iterator_setting(previous_iterator_setting ? previous_iterator_setting->c_str() : nullptr), 0);
    }

    bool enable_mmap() const {
        return std::get<1>(GetParam());
    }

    static std::filesystem::path model_path(const std::string& name) {
        return ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, name});
    }

    static std::string model_bytes(const std::string& name) {
        std::ifstream file(model_path(name), std::ios::binary);
        std::ostringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    InputModel::Ptr load_in_memory(const FrontEnd::Ptr& frontend, ModelProto& model_proto, bool as_proto) const {
        if (as_proto) {
            return frontend->load(reinterpret_cast<uint64_t>(&model_proto), enable_mmap());
        }
        std::istringstream stream(model_proto.SerializeAsString());
        return frontend->load(static_cast<std::istream*>(&stream), enable_mmap());
    }

    void check_progress(bool as_proto, bool as_iterator = false) {
        for (bool materialize_places : {false, true}) {
            for (int conversion_kind : {0, 1, 2}) {
                SCOPED_TRACE(testing::Message()
                             << "materialize_places=" << materialize_places << ", conversion_kind=" << conversion_kind);
                auto frontend = FrontEndManager().load_by_framework("onnx");
                std::vector<std::tuple<float, unsigned int, unsigned int>> progress;
                frontend->add_extension(std::make_shared<ProgressReporterExtension>(
                    [&](float fraction, unsigned int total, unsigned int completed) {
                        progress.emplace_back(fraction, total, completed);
                    }));
                ModelProto model_proto;
                ASSERT_TRUE(model_proto.ParseFromString(model_bytes("add_abc_initializers.onnx")));
                const auto expected_total = static_cast<unsigned int>(model_proto.graph().node_size());
                InputModel::Ptr input_model;
                if (as_iterator) {
                    auto iterator =
                        std::make_shared<ov::frontend::onnx::GraphIteratorProto>(ov::frontend::onnx::Internal_Stream);
                    iterator->initialize(std::make_shared<ModelProto>(model_proto));
                    iterator->reset();
                    EXPECT_EQ(iterator->size(), expected_total);
                    input_model = frontend->load(std::static_pointer_cast<ov::frontend::onnx::GraphIterator>(iterator));
                } else {
                    input_model = load_in_memory(frontend, model_proto, as_proto);
                }
                ASSERT_NE(input_model, nullptr);
                if (materialize_places) {
                    ASSERT_EQ(input_model->get_inputs().size(), 1);
                }
                if (conversion_kind == 0) {
                    frontend->convert(input_model);
                } else if (conversion_kind == 1) {
                    frontend->convert_partially(input_model);
                } else {
                    frontend->decode(input_model);
                }
                ASSERT_EQ(progress.size(), expected_total);
                float previous_fraction = 0.f;
                unsigned int previous_completed = 0;
                for (const auto& [fraction, total, completed] : progress) {
                    EXPECT_EQ(total, expected_total);
                    EXPECT_EQ(completed, previous_completed + 1);
                    EXPECT_LE(completed, total);
                    EXPECT_GE(fraction, previous_fraction);
                    EXPECT_FLOAT_EQ(fraction, static_cast<float>(completed) / total);
                    previous_fraction = fraction;
                    previous_completed = completed;
                }
                EXPECT_FLOAT_EQ(previous_fraction, 1.f);
                EXPECT_EQ(previous_completed, expected_total);
            }
        }
    }

    static void check_initializer_model(const std::shared_ptr<ov::Model>& model) {
        ASSERT_NE(model, nullptr);
        ASSERT_EQ(model->inputs().size(), 1);
        EXPECT_THAT(model->input().get_names(), UnorderedElementsAre("C"));
        EXPECT_THAT(model->output().get_names(), UnorderedElementsAre("Y"));
        ov::test::TestCase test_case(model);
        test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
        test_case.add_expected_output<float>(ov::Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});
        test_case.run();
    }
};

TEST_P(ONNXInMemoryLoadTest, selects_iterator_or_legacy_and_owns_stream_data) {
    FrontEndManager manager;
    auto frontend = std::make_shared<ov::frontend::onnx::FrontEnd>();
    InputModel::Ptr input_model;
    {
        std::istringstream stream(model_bytes("add_abc_initializers.onnx"));
        auto stream_ptr = static_cast<std::istream*>(&stream);
        ASSERT_NE(manager.load_by_model(stream_ptr), nullptr);
        ASSERT_TRUE(frontend->supported(stream_ptr));
        ASSERT_TRUE(frontend->supported(stream_ptr));
        EXPECT_EQ(stream.tellg(), std::streampos{0});
        input_model = frontend->load(stream_ptr, enable_mmap());
        ASSERT_NE(input_model, nullptr);
    }

    // Explicit GraphIterator input always uses the new importer, independently of ONNX_ITERATOR.
    auto iterator = std::make_shared<ov::frontend::onnx::GraphIteratorProto>(ov::frontend::onnx::Internal_Stream);
    iterator->initialize(model_path("add_abc_initializers.onnx"));
    iterator->reset();
    auto iterator_model = frontend->load(std::static_pointer_cast<ov::frontend::onnx::GraphIterator>(iterator));
    ASSERT_NE(iterator_model, nullptr);
    const auto& loaded_model = *input_model;
    const auto& explicit_iterator_model = *iterator_model;
    EXPECT_EQ(typeid(loaded_model) == typeid(explicit_iterator_model),
              ov::frontend::onnx::tests::is_graph_iterator_enabled());

    check_initializer_model(frontend->convert(input_model));
}

TEST_P(ONNXInMemoryLoadTest, loads_from_current_stream_position) {
    std::istringstream stream("prefix" + model_bytes("add_abc_initializers.onnx"));
    stream.seekg(6);
    auto frontend = FrontEndManager().load_by_framework("onnx");
    auto input_model = frontend->load(static_cast<std::istream*>(&stream), enable_mmap());
    ASSERT_NE(input_model, nullptr);
    check_initializer_model(frontend->convert(input_model));
}

TEST_P(ONNXInMemoryLoadTest, recovers_stream_at_eof) {
    std::istringstream stream(model_bytes("add_abc_initializers.onnx"));
    stream.seekg(0, std::ios::end);
    stream.peek();
    ASSERT_TRUE(stream.eof());
    auto frontend = FrontEndManager().load_by_framework("onnx");
    auto input_model = frontend->load(static_cast<std::istream*>(&stream), enable_mmap());
    ASSERT_NE(input_model, nullptr);
    check_initializer_model(frontend->convert(input_model));
}

TEST_P(ONNXInMemoryLoadTest, rejects_malformed_stream) {
    std::istringstream stream("not an ONNX protobuf");
    auto frontend = FrontEndManager().load_by_framework("onnx");
    OV_EXPECT_THROW(frontend->load(static_cast<std::istream*>(&stream), enable_mmap()),
                    ov::Exception,
                    testing::HasSubstr("Error during import of ONNX model provided as input stream"));
}

TEST_P(ONNXInMemoryLoadTest, decodes_stream_like_file) {
    auto frontend = FrontEndManager().load_by_framework("onnx");
    std::shared_ptr<ov::Model> decoded;
    {
        std::istringstream stream(model_bytes("add_abc_initializers.onnx"));
        auto input_model = frontend->load(static_cast<std::istream*>(&stream), enable_mmap());
        ASSERT_NE(input_model, nullptr);
        decoded = frontend->decode(input_model);
        ASSERT_NE(decoded, nullptr);
    }
    auto reference_input = frontend->load(model_path("add_abc_initializers.onnx"), enable_mmap());
    auto reference = frontend->decode(reference_input);
    auto comparator = FunctionsComparator::with_default()
                          .enable(FunctionsComparator::CONST_VALUES)
                          .enable(FunctionsComparator::ATTRIBUTES)
                          .enable(FunctionsComparator::TENSOR_NAMES);
    const auto result = comparator.compare(decoded, reference);
    EXPECT_TRUE(result.valid) << result.message;
}

TEST_P(ONNXInMemoryLoadTest, loads_external_weights_relative_to_stream_path) {
    auto frontend = FrontEndManager().load_by_framework("onnx");
    // The ONNX file need not exist: the supplied path only locates the external weights.
    const auto path = std::filesystem::path(model_path("external_data/stream_only.onnx"));
    for (const ov::Any& path_variant : {ov::Any(path.string()), ov::Any(path)}) {
        InputModel::Ptr input_model;
        {
            std::istringstream stream(model_bytes("external_data/external_data.onnx"));
            input_model = frontend->load(static_cast<std::istream*>(&stream), path_variant, enable_mmap());
            ASSERT_NE(input_model, nullptr);
        }
        auto model = frontend->convert(input_model);
        ASSERT_NE(model, nullptr);
        ov::test::TestCase test_case(model);
        test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
        test_case.add_expected_output<float>(ov::Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});
        test_case.run();
    }
}

TEST_P(ONNXInMemoryLoadTest, converts_control_flow_from_stream) {
    std::istringstream stream(model_bytes("controlflow/loop_2d_add.onnx"));
    auto frontend = FrontEndManager().load_by_framework("onnx");
    auto input_model = frontend->load(static_cast<std::istream*>(&stream), enable_mmap());
    ASSERT_NE(input_model, nullptr);
    auto model = frontend->convert(input_model);
    ASSERT_NE(model, nullptr);
    ov::test::TestCase test_case(model);
    test_case.add_input<float>({0.f, 0.f});
    test_case.add_expected_output<float>(ov::Shape{1, 2}, {3.f, 3.f});
    test_case.add_expected_output<float>(ov::Shape{3, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f});
    test_case.run();
}

TEST_P(ONNXInMemoryLoadTest, applies_conversion_extension_to_stream) {
    auto frontend = FrontEndManager().load_by_framework("onnx");
    frontend->add_extension(
        std::make_shared<ov::frontend::onnx::ConversionExtension>("Add", [](const ov::frontend::NodeContext& node) {
            return ov::OutputVector{std::make_shared<ov::op::v1::Multiply>(node.get_input(0), node.get_input(1))};
        }));
    std::istringstream stream(model_bytes("add_abc_initializers.onnx"));
    auto input_model = frontend->load(static_cast<std::istream*>(&stream), enable_mmap());
    ASSERT_NE(input_model, nullptr);
    auto model = frontend->convert(input_model);
    ASSERT_NE(model, nullptr);
    ov::test::TestCase test_case(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(ov::Shape{2, 2}, {1.f, 8.f, 27.f, 64.f});
    test_case.run();
}

TEST_P(ONNXInMemoryLoadTest, selects_iterator_or_legacy_and_copies_model_proto) {
    FrontEndManager manager;
    auto frontend = std::make_shared<ov::frontend::onnx::FrontEnd>();
    InputModel::Ptr input_model;
    {
        ModelProto model_proto;
        ASSERT_TRUE(model_proto.ParseFromString(model_bytes("add_abc_initializers.onnx")));
        const auto original = model_proto.SerializeAsString();
        const auto address = reinterpret_cast<uint64_t>(&model_proto);
        ASSERT_NE(manager.load_by_model(address), nullptr);
        ASSERT_TRUE(frontend->supported(address));
        input_model = frontend->load(address, enable_mmap());
        ASSERT_NE(input_model, nullptr);
        EXPECT_EQ(model_proto.SerializeAsString(), original);
        model_proto.Clear();
    }

    auto file_model = frontend->load(model_path("add_abc_initializers.onnx"), enable_mmap());
    ASSERT_NE(file_model, nullptr);
    const auto& loaded_model = *input_model;
    const auto& reference_model = *file_model;
    EXPECT_TRUE(typeid(loaded_model) == typeid(reference_model));
    check_initializer_model(frontend->convert(input_model));
}

TEST_P(ONNXInMemoryLoadTest, reuses_input_model_after_conversion) {
    for (bool as_proto : {false, true}) {
        SCOPED_TRACE(testing::Message() << "as_proto=" << as_proto);
        ModelProto model_proto;
        ASSERT_TRUE(model_proto.ParseFromString(model_bytes("add_abc_initializers.onnx")));
        auto frontend = FrontEndManager().load_by_framework("onnx");
        auto input_model = load_in_memory(frontend, model_proto, as_proto);
        ASSERT_NE(input_model, nullptr);

        check_initializer_model(frontend->convert(input_model));
        check_initializer_model(frontend->convert(input_model));

        const auto inputs = input_model->get_inputs();
        const auto outputs = input_model->get_outputs();
        ASSERT_EQ(inputs.size(), 1);
        ASSERT_EQ(outputs.size(), 1);
        EXPECT_THAT(inputs.front()->get_names(), ElementsAre("C"));
        EXPECT_THAT(outputs.front()->get_names(), ElementsAre("Y"));
        check_initializer_model(frontend->convert(input_model));
    }
}

TEST_P(ONNXInMemoryLoadTest, retries_lazy_conversion_after_error) {
    for (bool as_proto : {false, true}) {
        SCOPED_TRACE(testing::Message() << "as_proto=" << as_proto);
        ModelProto model_proto;
        ASSERT_TRUE(model_proto.ParseFromString(model_bytes("abs.onnx")));
        auto frontend = FrontEndManager().load_by_framework("onnx");
        bool fail_once = true;
        frontend->add_extension(std::make_shared<ov::frontend::onnx::ConversionExtension>(
            "Abs",
            [&fail_once](const ov::frontend::NodeContext& node) -> ov::OutputVector {
                if (fail_once) {
                    fail_once = false;
                    throw std::runtime_error("conversion failed once");
                }
                return {node.get_input(0)};
            }));
        auto input_model = load_in_memory(frontend, model_proto, as_proto);
        ASSERT_NE(input_model, nullptr);

        EXPECT_THROW(frontend->convert(input_model), std::runtime_error);
        const auto model = frontend->convert(input_model);
        ASSERT_NE(model, nullptr);
        EXPECT_EQ(model->inputs().size(), 1);
        EXPECT_THAT(model->output().get_names(), UnorderedElementsAre("y"));
    }
}

TEST_P(ONNXInMemoryLoadTest, rejects_invalid_model_proto) {
    auto frontend = FrontEndManager().load_by_framework("onnx");
    EXPECT_FALSE(frontend->supported(uint64_t{0}));
    OV_EXPECT_THROW(frontend->load(uint64_t{0}, enable_mmap()), ov::Exception, testing::HasSubstr("Wrong address"));

    ModelProto model_proto;
    ASSERT_TRUE(model_proto.ParseFromString(model_bytes("add_abc_initializers.onnx")));
    const auto address = reinterpret_cast<uint64_t>(&model_proto);
    model_proto.clear_ir_version();
    EXPECT_FALSE(frontend->supported(address));
    OV_EXPECT_THROW(frontend->load(address, enable_mmap()),
                    ov::Exception,
                    testing::HasSubstr("unsupported IR version"));

    model_proto.set_ir_version(Version::IR_VERSION + 1);
    EXPECT_FALSE(frontend->supported(address));
    OV_EXPECT_THROW(frontend->load(address, enable_mmap()),
                    ov::Exception,
                    testing::HasSubstr("unsupported IR version"));
}

TEST_P(ONNXInMemoryLoadTest, string_initializer_is_not_a_runtime_input) {
    ModelProto model_proto;
    ASSERT_TRUE(model_proto.ParseFromString(model_bytes("abs.onnx")));
    auto graph = model_proto.mutable_graph();
    auto initializer = graph->add_initializer();
    initializer->set_name("label");
    initializer->set_data_type(::ONNX_NAMESPACE::TensorProto_DataType_STRING);
    initializer->add_dims(1);
    initializer->add_string_data("constant label");
    auto input = graph->add_input();
    input->set_name("label");
    auto tensor_type = input->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType_STRING);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(1);

    auto frontend = FrontEndManager().load_by_framework("onnx");
    for (bool as_proto : {false, true}) {
        SCOPED_TRACE(testing::Message() << "as_proto=" << as_proto);
        auto input_model = load_in_memory(frontend, model_proto, as_proto);
        ASSERT_NE(input_model, nullptr);
        const auto inputs = input_model->get_inputs();
        ASSERT_EQ(inputs.size(), 1);
        EXPECT_THAT(inputs.front()->get_names(), ElementsAre("x"));
    }
}

TEST_P(ONNXInMemoryLoadTest, converts_control_flow_from_model_proto) {
    auto frontend = FrontEndManager().load_by_framework("onnx");
    InputModel::Ptr input_model;
    {
        ModelProto model_proto;
        ASSERT_TRUE(model_proto.ParseFromString(model_bytes("controlflow/loop_2d_add.onnx")));
        input_model = frontend->load(reinterpret_cast<uint64_t>(&model_proto), enable_mmap());
        ASSERT_NE(input_model, nullptr);
    }
    auto model = frontend->convert(input_model);
    ASSERT_NE(model, nullptr);
    ov::test::TestCase test_case(model);
    test_case.add_input<float>({0.f, 0.f});
    test_case.add_expected_output<float>(ov::Shape{1, 2}, {3.f, 3.f});
    test_case.add_expected_output<float>(ov::Shape{3, 1, 2}, {1.f, 1.f, 2.f, 2.f, 3.f, 3.f});
    test_case.run();
}

TEST_P(ONNXInMemoryLoadTest, reports_progress_from_stream) {
    check_progress(false);
}

TEST_P(ONNXInMemoryLoadTest, reports_progress_from_model_proto) {
    check_progress(true);
}

TEST_P(ONNXInMemoryLoadTest, reports_operation_progress_from_explicit_iterator) {
    check_progress(false, true);
}

TEST_P(ONNXInMemoryLoadTest, uses_default_for_undefined_optional_attribute) {
    for (const auto& attribute_mode : {"absent", "undefined", "defined"}) {
        for (bool as_proto : {false, true}) {
            for (bool materialize_places : {false, true}) {
                SCOPED_TRACE(testing::Message() << attribute_mode << ", as_proto=" << as_proto
                                                << ", materialize_places=" << materialize_places);
                ModelProto model_proto;
                ASSERT_TRUE(model_proto.ParseFromString(model_bytes("abs.onnx")));
                auto node = model_proto.mutable_graph()->mutable_node(0);
                node->set_op_type("LeakyRelu");
                const bool defined = std::string(attribute_mode) == "defined";
                if (std::string(attribute_mode) != "absent") {
                    auto attr = node->add_attribute();
                    attr->set_name("alpha");
                    attr->set_type(defined ? ::ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT
                                           : ::ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED);
                    if (defined) {
                        attr->set_f(0.5f);
                    }
                }
                auto frontend = FrontEndManager().load_by_framework("onnx");
                auto input_model = load_in_memory(frontend, model_proto, as_proto);
                ASSERT_NE(input_model, nullptr);
                if (materialize_places) {
                    ASSERT_EQ(input_model->get_inputs().size(), 1);
                }
                auto model = frontend->convert(input_model);
                ov::test::TestCase test_case(model);
                test_case.add_input<float>({-1.f, 2.f, -3.f});
                test_case.add_expected_output<float>(
                    ov::Shape{1, 3},
                    defined ? std::vector<float>{-0.5f, 2.f, -1.5f} : std::vector<float>{-0.01f, 2.f, -0.03f});
                test_case.run();
            }
        }
    }
}

TEST_P(ONNXInMemoryLoadTest, preserves_optimized_out_input_names) {
    for (const auto& op_type : {"Min", "Max", "Sum", "Identity"}) {
        for (bool as_proto : {false, true}) {
            for (bool materialize_places : {false, true}) {
                SCOPED_TRACE(testing::Message()
                             << op_type << ", as_proto=" << as_proto << ", materialize_places=" << materialize_places);
                ModelProto model_proto;
                ASSERT_TRUE(model_proto.ParseFromString(model_bytes("abs.onnx")));
                model_proto.mutable_graph()->mutable_node(0)->set_op_type(op_type);
                auto frontend = FrontEndManager().load_by_framework("onnx");
                auto input_model = load_in_memory(frontend, model_proto, as_proto);
                if (materialize_places) {
                    ASSERT_EQ(input_model->get_inputs().size(), 1);
                }
                auto model = frontend->convert(input_model);
                ASSERT_EQ(model->inputs().size(), 1);
                ASSERT_EQ(model->outputs().size(), 1);
                EXPECT_THAT(model->input().get_names(), Contains("x"));
                EXPECT_THAT(model->output().get_names(), Contains("y"));
            }
        }
    }
}

TEST_P(ONNXInMemoryLoadTest, preserves_zero_length_model_input) {
    for (bool as_proto : {false, true}) {
        for (bool materialize_places : {false, true}) {
            SCOPED_TRACE(testing::Message()
                         << "as_proto=" << as_proto << ", materialize_places=" << materialize_places);
            ModelProto model_proto;
            ASSERT_TRUE(model_proto.ParseFromString(model_bytes("abs.onnx")));
            auto graph = model_proto.mutable_graph();
            for (auto value_info : {graph->mutable_input(0), graph->mutable_output(0)}) {
                auto shape = value_info->mutable_type()->mutable_tensor_type()->mutable_shape();
                shape->clear_dim();
                shape->add_dim()->set_dim_value(0);
            }
            auto frontend = FrontEndManager().load_by_framework("onnx");
            auto input_model = load_in_memory(frontend, model_proto, as_proto);
            if (materialize_places) {
                ASSERT_EQ(input_model->get_inputs().size(), 1);
            }
            auto model = frontend->convert(input_model);
            ASSERT_EQ(model->inputs().size(), 1);
            EXPECT_EQ(model->input().get_shape(), ov::Shape{0});
            ov::test::TestCase test_case(model);
            test_case.add_input<float>(ov::Shape{0}, {});
            test_case.add_expected_output<float>(ov::Shape{0}, {});
            test_case.run();
        }
    }
}

TEST_P(ONNXInMemoryLoadTest, converts_loss_with_optional_ignore_index) {
    for (const auto& ignore_mode : {"absent", "undefined", "negative", "high"}) {
        for (bool as_proto : {false, true}) {
            SCOPED_TRACE(testing::Message() << ignore_mode << ", as_proto=" << as_proto);
            ModelProto model_proto;
            ASSERT_TRUE(model_proto.ParseFromString(model_bytes("negativelog_likelihood_loss.onnx")));
            auto node = model_proto.mutable_graph()->mutable_node(0);
            node->mutable_attribute(0)->set_s("sum");
            const bool ignore = std::string(ignore_mode) == "negative" || std::string(ignore_mode) == "high";
            const int64_t ignored_target = std::string(ignore_mode) == "negative" ? -1 : 5;
            if (std::string(ignore_mode) != "absent") {
                auto attr = node->add_attribute();
                attr->set_name("ignore_index");
                attr->set_type(ignore ? ::ONNX_NAMESPACE::AttributeProto_AttributeType_INT
                                      : ::ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED);
                if (ignore) {
                    attr->set_i(ignored_target);
                }
            }
            auto frontend = FrontEndManager().load_by_framework("onnx");
            auto input_model = load_in_memory(frontend, model_proto, as_proto);
            auto model = frontend->convert(input_model);
            ov::test::TestCase test_case(model);
            test_case.add_input<float>(std::vector<float>(30, -1.f));
            test_case.add_input<int64_t>({ignore ? ignored_target : 0, 1, 0, 1, 0, 1});
            test_case.add_expected_output<float>(ov::Shape{}, {ignore ? 5.f : 6.f});
            test_case.run();
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ONNX,
                         ONNXInMemoryLoadTest,
                         testing::Combine(testing::Values(static_cast<const char*>(nullptr), "1", "0"),
                                          testing::Bool()));

}  // namespace
