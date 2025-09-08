// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "load_from.hpp"

#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>

#include <fstream>

#include "common_test_utils/test_assertions.hpp"
#include "onnx_utils.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "utils.hpp"

using namespace ov::frontend;

using ONNXLoadTest = FrontEndLoadFromTest;
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
    const auto path =
        ov::util::path_join(
            {ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "external_data/external_data.onnx"})
            .string();
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
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, m_param.m_stream})
            .string();

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
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "abs.onnx"}).string();
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
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), TEST_ONNX_MODELS_DIRNAME, "abs.onnx"}).string();
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
