// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/manager.hpp>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/util.hpp"
#include "openvino/util/file_util.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    if defined(WINAPI_FAMILY) && !WINAPI_PARTITION_DESKTOP
#        error "Only WINAPI_PARTITION_DESKTOP is supported, because of LoadLibrary[A|W]"
#    endif
#elif defined(__linux) || defined(__APPLE__)
#    include <dlfcn.h>
#endif

#ifdef _WIN32
const char FrontEndPathSeparator[] = ";";
#else
const char FrontEndPathSeparator[] = ":";
#endif  // _WIN32

using namespace ngraph;
using namespace ov::frontend;

static std::string find_my_pathname() {
#ifdef _WIN32
    HMODULE hModule = GetModuleHandleW(SHARED_LIB_PREFIX L"openvino" SHARED_LIB_SUFFIX);
    WCHAR wpath[MAX_PATH];
    GetModuleFileNameW(hModule, wpath, MAX_PATH);
    std::wstring ws(wpath);
    std::string path(ws.begin(), ws.end());
    std::replace(path.begin(), path.end(), '\\', '/');
    NGRAPH_SUPPRESS_DEPRECATED_START
    path = file_util::get_directory(path);
    NGRAPH_SUPPRESS_DEPRECATED_END
    path += "/";
    return path;
#elif defined(__linux) || defined(__APPLE__)
    Dl_info dl_info;
    dladdr(reinterpret_cast<void*>(ngraph::to_lower), &dl_info);
    return ov::util::get_absolute_file_path(dl_info.dli_fname);
#else
#    error "Unsupported OS"
#endif
}

static int set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    return _putenv_s(name, value);
#elif defined(__linux) || defined(__APPLE__)
    std::string var = std::string(name) + "=" + value;
    return setenv(name, value, 0);
#endif
}

TEST(FrontEndManagerTest, testAvailableFrontEnds) {
    FrontEndManager fem;
    class MockFrontEnd : public FrontEnd {};
    ASSERT_NO_THROW(fem.register_front_end("mock", []() {
        return std::make_shared<MockFrontEnd>();
    }));
    auto frontends = fem.get_available_front_ends();
    ASSERT_NE(std::find(frontends.begin(), frontends.end(), "mock"), frontends.end());
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(fe = fem.load_by_framework("mock"));

    FrontEndManager fem2 = std::move(fem);
    frontends = fem2.get_available_front_ends();
    ASSERT_NE(std::find(frontends.begin(), frontends.end(), "mock"), frontends.end());

    fem2 = FrontEndManager();
    frontends = fem2.get_available_front_ends();
    ASSERT_EQ(std::find(frontends.begin(), frontends.end(), "mock"), frontends.end());
}

TEST(FrontEndManagerTest, testMockPluginFrontEnd) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    std::string fePath = ngraph::file_util::get_directory(find_my_pathname());
    fePath = fePath + FrontEndPathSeparator + "someInvalidPath";
    set_test_env("OV_FRONTEND_PATH", fePath.c_str());

    FrontEndManager fem;
    auto frontends = fem.get_available_front_ends();
    ASSERT_NE(std::find(frontends.begin(), frontends.end(), "mock1"), frontends.end());
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(fe = fem.load_by_framework("mock1"));
    ASSERT_EQ(fe->get_name(), "mock1");
    set_test_env("OV_FRONTEND_PATH", "");
    NGRAPH_SUPPRESS_DEPRECATED_END
}

TEST(FrontEndManagerTest, testFEMDestroy_FrontEndHolder) {
    FrontEnd::Ptr fe;
    {
        FrontEndManager fem;
        auto frontends = fem.get_available_front_ends();
        ASSERT_NE(std::find(frontends.begin(), frontends.end(), "mock1"), frontends.end());
        ASSERT_NO_THROW(fe = fem.load_by_framework("mock1"));
    }
    ASSERT_EQ(fe->get_name(), "mock1");
}

TEST(FrontEndManagerTest, testFEMDestroy_InputModelHolder) {
    InputModel::Ptr input_model;
    {
        std::shared_ptr<ov::Model> model;
        FrontEndManager fem;
        auto fe = fem.load_by_framework("mock1");
        input_model = fe->load("test");
        model = fe->convert(input_model);
        ASSERT_EQ(model->get_friendly_name(), "mock1_model");
    }
    ASSERT_TRUE(input_model);
}

TEST(FrontEndManagerTest, testFEMDestroy_OVModelHolder) {
    std::shared_ptr<ov::Model> model;
    {
        FrontEndManager fem;
        auto fe = fem.load_by_framework("mock1");
        auto input_model = fe->load("test");
        model = fe->convert(input_model);
        ASSERT_EQ(model->get_friendly_name(), "mock1_model");
        ASSERT_TRUE(model->get_rt_info().count("mock_test"));
        ASSERT_EQ(model->get_rt_info()["mock_test"].as<std::string>(), std::string(1024, 't'));
    }
    ASSERT_EQ(model->get_friendly_name(), "mock1_model");
}

TEST(FrontEndManagerTest, testFEMDestroy_OVModelHolder_Clone) {
    std::shared_ptr<ov::Model> model_clone;
    {
        FrontEndManager fem;
        auto fe = fem.load_by_framework("mock1");
        auto input_model = fe->load("test");
        auto model = fe->convert(input_model);
        ASSERT_EQ(model->get_friendly_name(), "mock1_model");
        ASSERT_TRUE(model->get_rt_info().count("mock_test"));
        ASSERT_EQ(model->get_rt_info()["mock_test"].as<std::string>(), std::string(1024, 't'));
        model_clone = ov::clone_model(*model);
    }
    ASSERT_EQ(model_clone->get_rt_info()["mock_test"].as<std::string>(), std::string(1024, 't'));
    ASSERT_EQ(model_clone->get_friendly_name(), "mock1_model");
}

TEST(FrontEndManagerTest, testDefaultFrontEnd) {
    FrontEndManager fem;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(fe = fem.load_by_model(""));
    ASSERT_FALSE(fe);

    class MockFrontEnd : public FrontEnd {};
    std::unique_ptr<FrontEnd> fePtr(new MockFrontEnd());  // to verify base destructor
    fe = std::make_shared<MockFrontEnd>();
    ASSERT_ANY_THROW(fe->load(""));
    ASSERT_ANY_THROW(fe->convert(std::shared_ptr<Function>(nullptr)));
    ASSERT_ANY_THROW(fe->convert(InputModel::Ptr(nullptr)));
    ASSERT_ANY_THROW(fe->convert_partially(nullptr));
    ASSERT_ANY_THROW(fe->decode(nullptr));
    ASSERT_ANY_THROW(fe->normalize(nullptr));
    ASSERT_EQ(fe->get_name(), std::string());
}

TEST(FrontEndManagerTest, testDefaultInputModel) {
    class MockInputModel : public InputModel {};
    std::unique_ptr<InputModel> imPtr(new MockInputModel());  // to verify base destructor
    InputModel::Ptr im = std::make_shared<MockInputModel>();
    ASSERT_EQ(im->get_inputs(), std::vector<Place::Ptr>{});
    ASSERT_EQ(im->get_outputs(), std::vector<Place::Ptr>{});
    ASSERT_ANY_THROW(im->override_all_inputs({nullptr}));
    ASSERT_ANY_THROW(im->override_all_outputs({nullptr}));
    ASSERT_ANY_THROW(im->extract_subgraph({nullptr}, {nullptr}));
    ASSERT_EQ(im->get_place_by_tensor_name(""), nullptr);
    ASSERT_EQ(im->get_place_by_operation_name(""), nullptr);
    ASSERT_EQ(im->get_place_by_operation_name_and_input_port("", 0), nullptr);
    ASSERT_EQ(im->get_place_by_operation_name_and_output_port("", 0), nullptr);
    ASSERT_ANY_THROW(im->set_name_for_tensor(nullptr, ""));
    ASSERT_ANY_THROW(im->add_name_for_tensor(nullptr, ""));
    ASSERT_ANY_THROW(im->set_name_for_operation(nullptr, ""));
    ASSERT_ANY_THROW(im->free_name_for_tensor(""));
    ASSERT_ANY_THROW(im->free_name_for_operation(""));
    ASSERT_ANY_THROW(im->set_name_for_dimension(nullptr, 0, ""));
    ASSERT_ANY_THROW(im->cut_and_add_new_input(nullptr, ""));
    ASSERT_ANY_THROW(im->cut_and_add_new_output(nullptr, ""));
    ASSERT_ANY_THROW(im->add_output(nullptr));
    ASSERT_ANY_THROW(im->remove_output(nullptr));
    ASSERT_ANY_THROW(im->set_partial_shape(nullptr, ngraph::Shape{}));
    ASSERT_ANY_THROW(im->get_partial_shape(nullptr));
    ASSERT_ANY_THROW(im->set_element_type(nullptr, ngraph::element::Type{}));
    ASSERT_ANY_THROW(im->set_tensor_value(nullptr, nullptr));
    ASSERT_ANY_THROW(im->set_tensor_partial_value(nullptr, nullptr, nullptr));
}

TEST(FrontEndManagerTest, testDefaultPlace) {
    class MockPlace : public Place {};
    std::unique_ptr<Place> placePtr(new MockPlace());  // to verify base destructor
    Place::Ptr place = std::make_shared<MockPlace>();
    ASSERT_ANY_THROW(place->get_names());
    ASSERT_EQ(place->get_consuming_operations(), std::vector<Place::Ptr>{});
    ASSERT_EQ(place->get_consuming_operations(0), std::vector<Place::Ptr>{});
    ASSERT_EQ(place->get_consuming_operations(""), std::vector<Place::Ptr>{});
    ASSERT_EQ(place->get_consuming_operations("", 0), std::vector<Place::Ptr>{});
    ASSERT_EQ(place->get_target_tensor(), nullptr);
    ASSERT_EQ(place->get_target_tensor(0), nullptr);
    ASSERT_EQ(place->get_target_tensor(""), nullptr);
    ASSERT_EQ(place->get_target_tensor("", 0), nullptr);
    ASSERT_EQ(place->get_source_tensor(), nullptr);
    ASSERT_EQ(place->get_source_tensor(""), nullptr);
    ASSERT_EQ(place->get_source_tensor(0), nullptr);
    ASSERT_EQ(place->get_source_tensor("", 0), nullptr);
    ASSERT_EQ(place->get_producing_operation(), nullptr);
    ASSERT_EQ(place->get_producing_operation(""), nullptr);
    ASSERT_EQ(place->get_producing_operation(0), nullptr);
    ASSERT_EQ(place->get_producing_operation("", 0), nullptr);
    ASSERT_EQ(place->get_producing_port(), nullptr);
    ASSERT_EQ(place->get_input_port(), nullptr);
    ASSERT_EQ(place->get_input_port(0), nullptr);
    ASSERT_EQ(place->get_input_port(""), nullptr);
    ASSERT_EQ(place->get_input_port("", 0), nullptr);
    ASSERT_EQ(place->get_output_port(), nullptr);
    ASSERT_EQ(place->get_output_port(0), nullptr);
    ASSERT_EQ(place->get_output_port(""), nullptr);
    ASSERT_EQ(place->get_output_port("", 0), nullptr);
    ASSERT_EQ(place->get_consuming_ports(), std::vector<Place::Ptr>{});
    ASSERT_ANY_THROW(place->is_input());
    ASSERT_ANY_THROW(place->is_output());
    ASSERT_ANY_THROW(place->is_equal(nullptr));
    ASSERT_ANY_THROW(place->is_equal_data(nullptr));
}

TEST(FrontEndExceptionTest, frontend_general_error_no_throw) {
    EXPECT_NO_THROW(FRONT_END_GENERAL_CHECK(true));
}

TEST(FrontEndExceptionTest, frontend_general_error_no_throw_info) {
    EXPECT_NO_THROW(FRONT_END_GENERAL_CHECK(true, "msg example"));
}

TEST(FrontEndExceptionTest, frontend_general_error_throw_no_info) {
    EXPECT_THROW(FRONT_END_GENERAL_CHECK(false), ov::frontend::GeneralFailure);
}

TEST(FrontEndExceptionTest, frontend_initialization_error_no_throw) {
    EXPECT_NO_THROW(FRONT_END_INITIALIZATION_CHECK(true));
}

TEST(FrontEndExceptionTest, frontend_initialization_error_no_throw_info) {
    EXPECT_NO_THROW(FRONT_END_INITIALIZATION_CHECK(true, "msg example"));
}

TEST(FrontEndExceptionTest, frontend_initialization_error_throw_no_info) {
    EXPECT_THROW(FRONT_END_INITIALIZATION_CHECK(false), ov::frontend::InitializationFailure);
}

TEST(FrontEndExceptionTest, frontend_op_conversion_error_no_throw) {
    EXPECT_NO_THROW(FRONT_END_OP_CONVERSION_CHECK(true));
}

TEST(FrontEndExceptionTest, frontend_op_conversion_error_no_throw_info) {
    EXPECT_NO_THROW(FRONT_END_OP_CONVERSION_CHECK(true, "msg example"));
}

TEST(FrontEndExceptionTest, frontend_op_conversion_error_throw_no_info) {
    EXPECT_THROW(FRONT_END_OP_CONVERSION_CHECK(false), ov::frontend::OpConversionFailure);
}

TEST(FrontEndExceptionTest, frontend_assert_throw_check_info) {
    std::string msg("msg example");
    try {
        FRONT_END_THROW(msg);
    } catch (const ov::frontend::GeneralFailure& ex) {
        std::string caught_msg(ex.what());
        EXPECT_NE(caught_msg.find(msg), std::string::npos);
        return;
    } catch (...) {
        FAIL() << "Not expected exception type.";
    }
    FAIL() << "Test is expected to throw an exception.";
}

TEST(FrontEndExceptionTest, frontend_not_implemented_throw_check_info) {
    struct TestClass {};
    try {
        FRONT_END_NOT_IMPLEMENTED(TestClass);
    } catch (const ov::frontend::NotImplementedFailure& ex) {
        std::string caught_msg(ex.what());
        EXPECT_NE(caught_msg.find("TestClass"), std::string::npos);
        return;
    } catch (...) {
        FAIL() << "Not expected exception type.";
    }
    FAIL() << "Test is expected to throw an exception.";
}

TEST(FrontEndExceptionTest, frontend_general_error_throw_info) {
    std::string msg("msg example");
    try {
        FRONT_END_GENERAL_CHECK(false, msg);
    } catch (const ov::frontend::GeneralFailure& ex) {
        std::string caught_msg(ex.what());
        EXPECT_NE(caught_msg.find(msg), std::string::npos);
        return;
    } catch (...) {
        FAIL() << "Not expected exception type.";
    }
    FAIL() << "Test is expected to throw an exception.";
}

TEST(FrontEndExceptionTest, frontend_op_conversion_error_throw_info) {
    std::string msg("msg example");
    try {
        FRONT_END_OP_CONVERSION_CHECK(false, msg);
    } catch (const ov::frontend::OpConversionFailure& ex) {
        std::string caught_msg(ex.what());
        EXPECT_NE(caught_msg.find(msg), std::string::npos);
        return;
    } catch (...) {
        FAIL() << "Not expected exception type.";
    }
    FAIL() << "Test is expected to throw an exception.";
}

TEST(FrontEndExceptionTest, frontend_initialization_error_throw_info) {
    std::string msg("msg example");
    try {
        FRONT_END_INITIALIZATION_CHECK(false, msg);
    } catch (const ov::frontend::InitializationFailure& ex) {
        std::string caught_msg(ex.what());
        EXPECT_NE(caught_msg.find(msg), std::string::npos);
        return;
    } catch (...) {
        FAIL() << "Not expected exception type.";
    }
    FAIL() << "Test is expected to throw an exception.";
}

// FrontEndManager exception safety
#define CHECK_EXCEPTION_FRONTEND(statement)                                                             \
    try {                                                                                               \
        FrontEndManager fem;                                                                            \
        auto fe = fem.load_by_framework("mock1");                                                       \
        auto input_model = fe->load("throw_next");                                                      \
        statement;                                                                                      \
        FAIL() << "Throw was expected";                                                                 \
    } catch (ov::frontend::GeneralFailure & error) {                                                    \
        EXPECT_NE(std::string(error.what()).find("Test exception"), std::string::npos) << error.what(); \
    } catch (...) {                                                                                     \
        FAIL() << "Unexpected error is thrown";                                                         \
    }

TEST(FrontEndManagerTest, Exception_Safety_FrontEnd_Load_By_Framework) {
    EXPECT_ANY_THROW({
        FrontEndManager fem;
        auto fe = fem.load_by_framework("mock1");
        fe->load("throw_now");
    });
}

TEST(FrontEndManagerTest, Exception_Safety_FrontEnd_Convert){CHECK_EXCEPTION_FRONTEND(fe->convert(input_model))}

TEST(FrontEndManagerTest,
     Exception_Safety_FrontEnd_Convert_OV_Model){CHECK_EXCEPTION_FRONTEND(fe->convert(std::shared_ptr<ov::Model>()))}

TEST(FrontEndManagerTest, Exception_Safety_FrontEnd_Get_Name){CHECK_EXCEPTION_FRONTEND(fe->get_name())}

TEST(FrontEndManagerTest, Exception_Safety_FrontEnd_Supported) {
    EXPECT_ANY_THROW({
        FrontEndManager fem;
        auto fe = fem.load_by_framework("mock1");
        fe->supported("throw_now");
    });
}

TEST(FrontEndManagerTest, Exception_Safety_FrontEnd_Add_Extension){
    CHECK_EXCEPTION_FRONTEND(fe->add_extension(std::make_shared<ov::Extension>()))}

TEST(FrontEndManagerTest,
     Exception_Safety_FrontEnd_Convert_Partially){CHECK_EXCEPTION_FRONTEND(fe->convert_partially(input_model))}

TEST(FrontEndManagerTest, Exception_Safety_FrontEnd_Normalize){CHECK_EXCEPTION_FRONTEND(fe->normalize(nullptr))}

TEST(FrontEndManagerTest, Exception_Safety_FrontEnd_Decode) {
    CHECK_EXCEPTION_FRONTEND(fe->decode(input_model))
}

// InputModel exception safety

#define CHECK_EXCEPTION_INPUT_MODEL(statement)                                                          \
    try {                                                                                               \
        FrontEndManager fem;                                                                            \
        auto fe = fem.load_by_framework("mock1");                                                       \
        auto input_model = fe->load("throw_model");                                                     \
        statement;                                                                                      \
        FAIL() << "Throw was expected";                                                                 \
    } catch (ov::frontend::GeneralFailure & error) {                                                    \
        EXPECT_NE(std::string(error.what()).find("Test exception"), std::string::npos) << error.what(); \
    } catch (...) {                                                                                     \
        FAIL() << "Unexpected error is thrown";                                                         \
    }

TEST(FrontEndManagerTest,
     Exception_Safety_Input_Model_get_inputs){CHECK_EXCEPTION_INPUT_MODEL(input_model->get_inputs())}

TEST(FrontEndManagerTest,
     Exception_Safety_Input_Model_get_outputs){CHECK_EXCEPTION_INPUT_MODEL(input_model->get_outputs())}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_get_place_by_tensor_name){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->get_place_by_tensor_name({}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_get_place_by_operation_name){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->get_place_by_operation_name({}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_get_place_by_operation_name_and_input_port){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->get_place_by_operation_name_and_input_port({}, {}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_get_place_by_operation_name_and_output_port){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->get_place_by_operation_name_and_output_port({}, {}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_set_name_for_tensor){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->set_name_for_tensor({}, {}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_add_name_for_tensor){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->add_name_for_tensor({}, {}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_set_name_for_operation){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->set_name_for_operation({}, {}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_free_name_for_tensor){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->free_name_for_tensor({}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_free_name_for_operation){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->free_name_for_operation({}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_set_name_for_dimension){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->set_name_for_dimension({}, {}, {}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_cut_and_add_new_input){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->cut_and_add_new_input({}, {}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_cut_and_add_new_output){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->cut_and_add_new_output({}, {}))}

TEST(FrontEndManagerTest,
     Exception_Safety_Input_Model_add_output){CHECK_EXCEPTION_INPUT_MODEL(input_model->add_output({}))}

TEST(FrontEndManagerTest,
     Exception_Safety_Input_Model_remove_output){CHECK_EXCEPTION_INPUT_MODEL(input_model->remove_output({}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_override_all_outputs){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->override_all_outputs({}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_override_all_inputs){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->override_all_inputs({}))}

TEST(FrontEndManagerTest,
     Exception_Safety_Input_Model_extract_subgraph){CHECK_EXCEPTION_INPUT_MODEL(input_model->extract_subgraph({}, {}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_set_partial_shape){
    CHECK_EXCEPTION_INPUT_MODEL(input_model->set_partial_shape({}, {}))}

TEST(FrontEndManagerTest,
     Exception_Safety_Input_Model_get_partial_shape){CHECK_EXCEPTION_INPUT_MODEL(input_model->get_partial_shape({}))}

TEST(FrontEndManagerTest,
     Exception_Safety_Input_Model_set_element_type){CHECK_EXCEPTION_INPUT_MODEL(input_model->set_element_type({}, {}))}

TEST(FrontEndManagerTest,
     Exception_Safety_Input_Model_set_tensor_value){CHECK_EXCEPTION_INPUT_MODEL(input_model->set_tensor_value({}, {}))}

TEST(FrontEndManagerTest, Exception_Safety_Input_Model_set_tensor_partial_value) {
    CHECK_EXCEPTION_INPUT_MODEL(input_model->set_tensor_partial_value({}, {}, {}))
}
