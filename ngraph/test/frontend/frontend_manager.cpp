// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    ASSERT_NO_THROW(fem.register_front_end(
        "mock", [](FrontEndCapFlags fec) { return std::make_shared<FrontEnd>(); }));
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

TEST(FrontEndManagerTest, testLoadWithFlags)
{
    int expFlags = FrontEndCapabilities::FEC_CUT | FrontEndCapabilities::FEC_WILDCARDS |
                   FrontEndCapabilities::FEC_NAMES;
    int actualFlags = FrontEndCapabilities::FEC_DEFAULT;
    FrontEndManager fem;
    ASSERT_NO_THROW(fem.register_front_end("mock", [&actualFlags](int fec) {
        actualFlags = fec;
        return std::make_shared<FrontEnd>();
    }));
    auto frontends = fem.get_available_front_ends();
    ASSERT_NE(std::find(frontends.begin(), frontends.end(), "mock"), frontends.end());
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(fe = fem.load_by_framework("mock", expFlags));
    ASSERT_TRUE(actualFlags & FrontEndCapabilities::FEC_CUT);
    ASSERT_TRUE(actualFlags & FrontEndCapabilities::FEC_WILDCARDS);
    ASSERT_TRUE(actualFlags & FrontEndCapabilities::FEC_NAMES);
    ASSERT_EQ(expFlags, actualFlags);
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
    ASSERT_ANY_THROW(fem.load_by_model(""));

    std::unique_ptr<FrontEnd> fePtr(new FrontEnd()); // to verify base destructor
    FrontEnd::Ptr fe = std::make_shared<FrontEnd>();
    ASSERT_ANY_THROW(fe->load_from_file(""));
    ASSERT_ANY_THROW(fe->load_from_files({"", ""}));
    ASSERT_ANY_THROW(fe->load_from_memory(nullptr));
    ASSERT_ANY_THROW(fe->load_from_memory_fragments({nullptr, nullptr}));
    std::stringstream str;
    ASSERT_ANY_THROW(fe->load_from_stream(str));
    ASSERT_ANY_THROW(fe->load_from_streams({&str, &str}));
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
