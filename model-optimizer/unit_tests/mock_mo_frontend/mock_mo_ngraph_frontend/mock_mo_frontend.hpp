// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "frontend_manager/frontend_manager.hpp"
#include "frontend_manager/frontend_manager_defs.hpp"
#include "ngraph/visibility.hpp"

// Defined if we are building the plugin DLL (instead of using it)
#ifdef mock_mo_ngraph_frontend_EXPORTS
#define MOCK_API NGRAPH_HELPER_DLL_EXPORT
#else
#define MOCK_API NGRAPH_HELPER_DLL_IMPORT
#endif // mock_mo_ngraph_frontend_EXPORTS

// OK to have 'using' in mock header

using namespace ngraph;
using namespace ngraph::frontend;

////////////////////////////////

/// \brief This structure holds number of calls of particular methods of Place objects
/// It will be used by Python unit tests to verify that appropriate API
/// was called with correct arguments during test execution
struct MOCK_API PlaceStat
{
    int m_get_names = 0;
    int m_get_input_port = 0;
    int m_get_output_port = 0;
    int m_is_input = 0;
    int m_is_output = 0;
    int m_is_equal = 0;

    // Arguments tracking
    std::string m_lastArgString;
    int m_lastArgInt;
    Place::Ptr m_lastArgPlace = nullptr;

    // Getters
    int get_names() const { return m_get_names; }
    int get_input_port() const { return m_get_input_port; }
    int get_output_port() const { return m_get_output_port; }
    int is_input() const { return m_is_input; }
    int is_output() const { return m_is_output; }
    int is_equal() const { return m_is_equal; }

    // Arguments getters
    std::string get_lastArgString() const { return m_lastArgString; }
    int get_lastArgInt() const { return m_lastArgInt; }
    Place::Ptr get_lastArgPlace() const { return m_lastArgPlace; }
};

/// \brief Mock implementation of Place
/// Every call increments appropriate counters in statistic and stores argument values to statistics
/// as well
class MOCK_API PlaceMockPy : public Place
{
    static PlaceStat m_stat;
    std::string m_name;

public:
    PlaceMockPy(const std::string& name = {})
        : m_name(name)
    {
    }

    std::vector<std::string> get_names() const override
    {
        m_stat.m_get_names++;
        return {m_name};
    }

    Place::Ptr get_input_port() const override
    {
        m_stat.m_get_input_port++;
        m_stat.m_lastArgInt = -1;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_input_port(int inputPortIndex) const override
    {
        m_stat.m_get_input_port++;
        m_stat.m_lastArgInt = inputPortIndex;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_input_port(const std::string& inputName) const override
    {
        m_stat.m_get_input_port++;
        m_stat.m_lastArgInt = -1;
        m_stat.m_lastArgString = inputName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_input_port(const std::string& inputName, int inputPortIndex) const override
    {
        m_stat.m_get_input_port++;
        m_stat.m_lastArgInt = inputPortIndex;
        m_stat.m_lastArgString = inputName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_output_port() const override
    {
        m_stat.m_get_output_port++;
        m_stat.m_lastArgInt = -1;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_output_port(int outputPortIndex) const override
    {
        m_stat.m_get_output_port++;
        m_stat.m_lastArgInt = outputPortIndex;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_output_port(const std::string& outputName) const override
    {
        m_stat.m_get_output_port++;
        m_stat.m_lastArgInt = -1;
        m_stat.m_lastArgString = outputName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_output_port(const std::string& outputName, int outputPortIndex) const override
    {
        m_stat.m_get_output_port++;
        m_stat.m_lastArgInt = outputPortIndex;
        m_stat.m_lastArgString = outputName;
        return std::make_shared<PlaceMockPy>();
    }

    bool is_input() const override
    {
        m_stat.m_is_input++;
        return m_name.find("input") != std::string::npos;
    }

    bool is_output() const override
    {
        m_stat.m_is_output++;
        return m_name.find("output") != std::string::npos;
    }

    bool is_equal(Ptr another) const override
    {
        m_stat.m_is_equal++;
        m_stat.m_lastArgPlace = another;
        return m_name == another->get_names().at(0);
    }

    //---------------Stat--------------------
    static PlaceStat get_stat() { return m_stat; }
    static void clear_stat() { m_stat = {}; }
};

////////////////////////////////

/// \brief This structure holds number of calls of particular methods of InputModel objects
/// It will be used by Python unit tests to verify that appropriate API
/// was called with correct arguments during test execution
struct MOCK_API ModelStat
{
    int m_get_inputs = 0;
    int m_get_outputs = 0;
    int m_get_place_by_tensor_name = 0;
    int m_set_partial_shape = 0;
    int m_get_partial_shape = 0;
    int m_set_element_type = 0;

    int m_extract_subgraph = 0;
    int m_override_all_inputs = 0;
    int m_override_all_outputs = 0;

    // Arguments tracking
    std::string m_lastArgString;
    int m_lastArgInt;
    Place::Ptr m_lastArgPlace = nullptr;
    std::vector<Place::Ptr> m_lastArgInputPlaces;
    std::vector<Place::Ptr> m_lastArgOutputPlaces;
    ngraph::element::Type m_lastArgElementType;
    ngraph::PartialShape m_lastArgPartialShape;

    // Getters
    int get_inputs() const { return m_get_inputs; }
    int get_outputs() const { return m_get_outputs; }
    int extract_subgraph() const { return m_extract_subgraph; }
    int override_all_inputs() const { return m_override_all_inputs; }
    int override_all_outputs() const { return m_override_all_outputs; }
    int get_place_by_tensor_name() const { return m_get_place_by_tensor_name; }
    int set_partial_shape() const { return m_set_partial_shape; }
    int get_partial_shape() const { return m_get_partial_shape; }
    int set_element_type() const { return m_set_element_type; }

    // Arguments getters
    std::string get_lastArgString() const { return m_lastArgString; }
    int get_lastArgInt() const { return m_lastArgInt; }
    Place::Ptr get_lastArgPlace() const { return m_lastArgPlace; }
    std::vector<Place::Ptr> get_lastArgInputPlaces() const { return m_lastArgInputPlaces; }
    std::vector<Place::Ptr> get_lastArgOutputPlaces() const { return m_lastArgOutputPlaces; }
    ngraph::element::Type get_lastArgElementType() const { return m_lastArgElementType; }
    ngraph::PartialShape get_lastArgPartialShape() const { return m_lastArgPartialShape; }
};

/// \brief Mock implementation of InputModel
/// Every call increments appropriate counters in statistic and stores argument values to statistics
/// as well
/// ("mock_output1", "mock_output2")
class MOCK_API InputModelMockPy : public InputModel
{
    static ModelStat m_stat;
    static PartialShape m_returnShape;

public:
    std::vector<Place::Ptr> get_inputs() const override
    {
        m_stat.m_get_inputs++;
        return {std::make_shared<PlaceMockPy>("mock_input1"),
                std::make_shared<PlaceMockPy>("mock_input2")};
    }

    std::vector<Place::Ptr> get_outputs() const override
    {
        m_stat.m_get_outputs++;
        return {std::make_shared<PlaceMockPy>("mock_output1"),
                std::make_shared<PlaceMockPy>("mock_output2")};
    }

    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override
    {
        m_stat.m_get_place_by_tensor_name++;
        m_stat.m_lastArgString = tensorName;
        return std::make_shared<PlaceMockPy>(tensorName);
    }

    void override_all_outputs(const std::vector<Place::Ptr>& outputs) override
    {
        m_stat.m_override_all_outputs++;
        m_stat.m_lastArgOutputPlaces = outputs;
    }

    void override_all_inputs(const std::vector<Place::Ptr>& inputs) override
    {
        m_stat.m_override_all_inputs++;
        m_stat.m_lastArgInputPlaces = inputs;
    }

    void extract_subgraph(const std::vector<Place::Ptr>& inputs,
                          const std::vector<Place::Ptr>& outputs) override
    {
        m_stat.m_extract_subgraph++;
        m_stat.m_lastArgInputPlaces = inputs;
        m_stat.m_lastArgOutputPlaces = outputs;
    }

    // Setting tensor properties
    void set_partial_shape(Place::Ptr place, const ngraph::PartialShape& shape) override
    {
        m_stat.m_set_partial_shape++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgPartialShape = shape;
    }

    ngraph::PartialShape get_partial_shape(Place::Ptr place) const override
    {
        m_stat.m_get_partial_shape++;
        m_stat.m_lastArgPlace = place;
        return m_returnShape;
    }

    void set_element_type(Place::Ptr place, const ngraph::element::Type& type) override
    {
        m_stat.m_set_element_type++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgElementType = type;
    }

    static void mock_return_partial_shape(const PartialShape& shape) { m_returnShape = shape; }

    //---------------Stat--------------------
    static ModelStat get_stat() { return m_stat; }
    static void clear_stat() { m_stat = {}; }
};

/////////////////////////////////////////////////////////

/// \brief This structure holds number of calls of particular methods of FrontEnd objects
/// It will be used by Python unit tests to verify that appropriate API
/// was called with correct arguments during test execution
struct MOCK_API FeStat
{
    std::vector<std::string> m_load_paths;
    int m_convert_model = 0;
    // Getters
    std::vector<std::string> load_paths() const { return m_load_paths; }
    int convert_model() const { return m_convert_model; }
};

/// \brief Mock implementation of FrontEnd
/// Every call increments appropriate counters in statistic and stores argument values to statistics
/// as well
class MOCK_API FrontEndMockPy : public FrontEnd
{
    static FeStat m_stat;

public:
    FrontEndMockPy() {}


    std::shared_ptr<ngraph::Function> convert(InputModel::Ptr model) const override
    {
        m_stat.m_convert_model++;
        return std::make_shared<ngraph::Function>(NodeVector{}, ParameterVector{});
    }

    static FeStat get_stat() { return m_stat; }

    static void clear_stat() { m_stat = {}; }

protected:
    InputModel::Ptr load_impl(const std::vector<std::shared_ptr<Variant>>& params) const override
    {
        if (params.size() > 0 && is_type<VariantWrapper<std::string>>(params[0]))
        {
            auto path = as_type_ptr<VariantWrapper<std::string>>(params[0])->get();
            m_stat.m_load_paths.push_back(path);
        }
        return std::make_shared<InputModelMockPy>();
    }
};
