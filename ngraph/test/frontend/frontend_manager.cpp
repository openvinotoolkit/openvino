// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <frontend_manager/frontend_exceptions.hpp>
#include <frontend_manager/frontend_manager.hpp>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "backend.hpp"
#include "ngraph/file_util.hpp"

#ifdef _WIN32
const char FrontEndPathSeparator[] = ";";
#else
const char FrontEndPathSeparator[] = ":";
#endif // _WIN32

using namespace ngraph;
using namespace ngraph::frontend;

static int set_test_env(const char* name, const char* value)
{
#ifdef _WIN32
    return _putenv_s(name, value);
#elif defined(__linux) || defined(__APPLE__)
    std::string var = std::string(name) + "=" + value;
    return setenv(name, value, 0);
#endif
}

TEST(FrontEndManagerTest, testAvailableFrontEnds)
{
    FrontEndManager fem;
    ASSERT_NO_THROW(fem.register_front_end("mock", []() { return std::make_shared<FrontEnd>(); }));
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

TEST(FrontEndManagerTest, testMockPluginFrontEnd)
{
    std::string fePath = ngraph::file_util::get_directory(
        ngraph::runtime::Backend::get_backend_shared_library_search_directory());
    fePath = fePath + FrontEndPathSeparator + "someInvalidPath";
    set_test_env("OV_FRONTEND_PATH", fePath.c_str());

    FrontEndManager fem;
    auto frontends = fem.get_available_front_ends();
    ASSERT_NE(std::find(frontends.begin(), frontends.end(), "mock1"), frontends.end());
    set_test_env("OV_FRONTEND_PATH", "");
}

TEST(FrontEndManagerTest, testDefaultFrontEnd)
{
    FrontEndManager fem;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(fe = fem.load_by_model(""));
    ASSERT_FALSE(fe);

    std::unique_ptr<FrontEnd> fePtr(new FrontEnd()); // to verify base destructor
    fe = std::make_shared<FrontEnd>();
    ASSERT_ANY_THROW(fe->load(""));
    ASSERT_ANY_THROW(fe->convert(std::shared_ptr<Function>(nullptr)));
    ASSERT_ANY_THROW(fe->convert(InputModel::Ptr(nullptr)));
    ASSERT_ANY_THROW(fe->convert_partially(nullptr));
    ASSERT_ANY_THROW(fe->decode(nullptr));
    ASSERT_ANY_THROW(fe->normalize(nullptr));
}

TEST(FrontEndManagerTest, testDefaultInputModel)
{
    std::unique_ptr<InputModel> imPtr(new InputModel()); // to verify base destructor
    InputModel::Ptr im = std::make_shared<InputModel>();
    ASSERT_ANY_THROW(im->get_inputs());
    ASSERT_ANY_THROW(im->get_outputs());
    ASSERT_ANY_THROW(im->override_all_inputs({nullptr}));
    ASSERT_ANY_THROW(im->override_all_outputs({nullptr}));
    ASSERT_ANY_THROW(im->extract_subgraph({nullptr}, {nullptr}));
    ASSERT_ANY_THROW(im->get_place_by_tensor_name(""));
    ASSERT_ANY_THROW(im->get_place_by_operation_name(""));
    ASSERT_ANY_THROW(im->get_place_by_operation_name_and_input_port("", 0));
    ASSERT_ANY_THROW(im->get_place_by_operation_name_and_output_port("", 0));
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

TEST(FrontEndManagerTest, testDefaultPlace)
{
    std::unique_ptr<Place> placePtr(new Place()); // to verify base destructor
    Place::Ptr place = std::make_shared<Place>();
    ASSERT_ANY_THROW(place->get_names());
    ASSERT_ANY_THROW(place->get_consuming_operations());
    ASSERT_ANY_THROW(place->get_consuming_operations(0));
    ASSERT_ANY_THROW(place->get_target_tensor());
    ASSERT_ANY_THROW(place->get_target_tensor(0));
    ASSERT_ANY_THROW(place->get_source_tensor());
    ASSERT_ANY_THROW(place->get_source_tensor(0));
    ASSERT_ANY_THROW(place->get_producing_operation());
    ASSERT_ANY_THROW(place->get_producing_operation(0));
    ASSERT_ANY_THROW(place->get_producing_port());
    ASSERT_ANY_THROW(place->get_input_port());
    ASSERT_ANY_THROW(place->get_input_port(0));
    ASSERT_ANY_THROW(place->get_input_port(""));
    ASSERT_ANY_THROW(place->get_input_port("", 0));
    ASSERT_ANY_THROW(place->get_output_port());
    ASSERT_ANY_THROW(place->get_output_port(0));
    ASSERT_ANY_THROW(place->get_output_port(""));
    ASSERT_ANY_THROW(place->get_output_port("", 0));
    ASSERT_ANY_THROW(place->get_consuming_ports());
    ASSERT_ANY_THROW(place->is_input());
    ASSERT_ANY_THROW(place->is_output());
    ASSERT_ANY_THROW(place->is_equal(nullptr));
    ASSERT_ANY_THROW(place->is_equal_data(nullptr));
}

TEST(FrontEndExceptionTest, frontend_general_error_no_throw)
{
    EXPECT_NO_THROW(FRONT_END_GENERAL_CHECK(true));
}

TEST(FrontEndExceptionTest, frontend_general_error_no_throw_info)
{
    EXPECT_NO_THROW(FRONT_END_GENERAL_CHECK(true, "msg example"));
}

TEST(FrontEndExceptionTest, frontend_general_error_throw_no_info)
{
    EXPECT_THROW(FRONT_END_GENERAL_CHECK(false), ngraph::frontend::GeneralFailure);
}

TEST(FrontEndExceptionTest, frontend_initialization_error_no_throw)
{
    EXPECT_NO_THROW(FRONT_END_INITIALIZATION_CHECK(true));
}

TEST(FrontEndExceptionTest, frontend_initialization_error_no_throw_info)
{
    EXPECT_NO_THROW(FRONT_END_INITIALIZATION_CHECK(true, "msg example"));
}

TEST(FrontEndExceptionTest, frontend_initialization_error_throw_no_info)
{
    EXPECT_THROW(FRONT_END_INITIALIZATION_CHECK(false), ngraph::frontend::InitializationFailure);
}

TEST(FrontEndExceptionTest, frontend_op_conversion_error_no_throw)
{
    EXPECT_NO_THROW(FRONT_END_OP_CONVERSION_CHECK(true));
}

TEST(FrontEndExceptionTest, frontend_op_conversion_error_no_throw_info)
{
    EXPECT_NO_THROW(FRONT_END_OP_CONVERSION_CHECK(true, "msg example"));
}

TEST(FrontEndExceptionTest, frontend_op_conversion_error_throw_no_info)
{
    EXPECT_THROW(FRONT_END_OP_CONVERSION_CHECK(false), ngraph::frontend::OpConversionFailure);
}

TEST(FrontEndExceptionTest, frontend_assert_throw_check_info)
{
    std::string msg("msg example");
    try
    {
        FRONT_END_THROW(msg);
    }
    catch (const ngraph::frontend::GeneralFailure& ex)
    {
        std::string caught_msg(ex.what());
        EXPECT_NE(caught_msg.find(msg), std::string::npos);
        return;
    }
    catch (...)
    {
        FAIL() << "Not expected exception type.";
    }
    FAIL() << "Test is expected to throw an exception.";
}

TEST(FrontEndExceptionTest, frontend_not_implemented_throw_check_info)
{
    struct TestClass
    {
    };
    try
    {
        FRONT_END_NOT_IMPLEMENTED(TestClass);
    }
    catch (const ngraph::frontend::NotImplementedFailure& ex)
    {
        std::string caught_msg(ex.what());
        EXPECT_NE(caught_msg.find("TestClass"), std::string::npos);
        return;
    }
    catch (...)
    {
        FAIL() << "Not expected exception type.";
    }
    FAIL() << "Test is expected to throw an exception.";
}

TEST(FrontEndExceptionTest, frontend_general_error_throw_info)
{
    std::string msg("msg example");
    try
    {
        FRONT_END_GENERAL_CHECK(false, msg);
    }
    catch (const ngraph::frontend::GeneralFailure& ex)
    {
        std::string caught_msg(ex.what());
        EXPECT_NE(caught_msg.find(msg), std::string::npos);
        return;
    }
    catch (...)
    {
        FAIL() << "Not expected exception type.";
    }
    FAIL() << "Test is expected to throw an exception.";
}

TEST(FrontEndExceptionTest, frontend_op_conversion_error_throw_info)
{
    std::string msg("msg example");
    try
    {
        FRONT_END_OP_CONVERSION_CHECK(false, msg);
    }
    catch (const ngraph::frontend::OpConversionFailure& ex)
    {
        std::string caught_msg(ex.what());
        EXPECT_NE(caught_msg.find(msg), std::string::npos);
        return;
    }
    catch (...)
    {
        FAIL() << "Not expected exception type.";
    }
    FAIL() << "Test is expected to throw an exception.";
}

TEST(FrontEndExceptionTest, frontend_initialization_error_throw_info)
{
    std::string msg("msg example");
    try
    {
        FRONT_END_INITIALIZATION_CHECK(false, msg);
    }
    catch (const ngraph::frontend::InitializationFailure& ex)
    {
        std::string caught_msg(ex.what());
        EXPECT_NE(caught_msg.find(msg), std::string::npos);
        return;
    }
    catch (...)
    {
        FAIL() << "Not expected exception type.";
    }
    FAIL() << "Test is expected to throw an exception.";
}
