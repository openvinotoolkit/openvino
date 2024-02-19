// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/visibility.hpp"

// Defined if we are building the plugin DLL (instead of using it)
#ifdef openvino_mock_mo_frontend_EXPORTS
#define MOCK_API OPENVINO_CORE_EXPORTS
#else
#define MOCK_API OPENVINO_CORE_IMPORTS
#endif // openvino_mock_mo_frontend_EXPORTS

// OK to have 'using' in mock header

using namespace ov::frontend;

////////////////////////////////
/// \brief This structure holds number static setup values
/// It will be used by Python unit tests to setup particular mock behavior
struct MOCK_API MockSetup
{
    static std::string m_equal_data_node1;
    static std::string m_equal_data_node2;
    static int m_max_input_port_index;
    static int m_max_output_port_index;

    static void clear_setup()
    {
        m_equal_data_node1 = {};
        m_equal_data_node2 = {};
        m_max_input_port_index = 0;
        m_max_output_port_index = 0;
    }

    static void set_equal_data(const std::string& node1, const std::string& node2)
    {
        m_equal_data_node1 = node1;
        m_equal_data_node2 = node2;
    }

    static void set_max_port_counts(int max_input, int max_output)
    {
        m_max_input_port_index = max_input;
        m_max_output_port_index = max_output;
    }
};

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
    int m_is_equal_data = 0;

    // Arguments tracking
    std::string m_lastArgString;
    int m_lastArgInt = -1;
    Place::Ptr m_lastArgPlace = nullptr;

    // Getters
    int get_names() const { return m_get_names; }
    int get_input_port() const { return m_get_input_port; }
    int get_output_port() const { return m_get_output_port; }
    int is_input() const { return m_is_input; }
    int is_output() const { return m_is_output; }
    int is_equal() const { return m_is_equal; }
    int is_equal_data() const { return m_is_equal_data; }

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
    bool m_is_op = false;
    int m_portIndex = -1;

public:
    explicit PlaceMockPy(std::string name = {}, bool is_op = false, int portIndex = -1)
        : m_name(std::move(name))
        , m_is_op(is_op)
        , m_portIndex(portIndex)
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
        if (inputPortIndex < MockSetup::m_max_input_port_index)
        {
            return std::make_shared<PlaceMockPy>(m_name, false, inputPortIndex);
        }
        return nullptr;
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
        if (outputPortIndex < MockSetup::m_max_output_port_index)
        {
            return std::make_shared<PlaceMockPy>(m_name, false, outputPortIndex);
        }
        return nullptr;
    }

    Place::Ptr get_output_port(const std::string& outputName) const override
    {
        m_stat.m_get_output_port++;
        m_stat.m_lastArgInt = -1;
        m_stat.m_lastArgString = outputName;
        return std::make_shared<PlaceMockPy>(outputName);
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

    bool is_equal(const Ptr& another) const override
    {
        m_stat.m_is_equal++;
        m_stat.m_lastArgPlace = another;
        std::shared_ptr<PlaceMockPy> mock = std::dynamic_pointer_cast<PlaceMockPy>(another);
        return m_name == mock->m_name && m_is_op == mock->m_is_op &&
               m_portIndex == mock->m_portIndex;
    }

    bool is_equal_data(const Ptr& another) const override
    {
        if (m_is_op)
            throw std::runtime_error("Not implemented");
        m_stat.m_is_equal_data++;
        m_stat.m_lastArgPlace = another;
        std::shared_ptr<PlaceMockPy> mock = std::dynamic_pointer_cast<PlaceMockPy>(another);
        if (!MockSetup::m_equal_data_node1.empty() && !MockSetup::m_equal_data_node2.empty())
        {
            if ((mock->m_name.find(MockSetup::m_equal_data_node1) != std::string::npos ||
                 mock->m_name.find(MockSetup::m_equal_data_node2) != std::string::npos) &&
                (m_name.find(MockSetup::m_equal_data_node1) != std::string::npos ||
                 m_name.find(MockSetup::m_equal_data_node2) != std::string::npos))
            {
                return true;
            }
        }
        return !mock->m_is_op && m_name == mock->m_name;
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
    int m_get_place_by_operation_name = 0;
    int m_set_partial_shape = 0;
    int m_get_partial_shape = 0;
    int m_set_element_type = 0;

    int m_extract_subgraph = 0;
    int m_override_all_inputs = 0;
    int m_override_all_outputs = 0;

    // Arguments tracking
    std::string m_lastArgString;
    int m_lastArgInt = -1;
    Place::Ptr m_lastArgPlace = nullptr;
    std::vector<Place::Ptr> m_lastArgInputPlaces;
    std::vector<Place::Ptr> m_lastArgOutputPlaces;
    ov::element::Type m_lastArgElementType;
    ov::PartialShape m_lastArgPartialShape;

    // Getters
    int get_inputs() const { return m_get_inputs; }
    int get_outputs() const { return m_get_outputs; }
    int extract_subgraph() const { return m_extract_subgraph; }
    int override_all_inputs() const { return m_override_all_inputs; }
    int override_all_outputs() const { return m_override_all_outputs; }
    int get_place_by_operation_name() const { return m_get_place_by_operation_name; }
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
    ov::element::Type get_lastArgElementType() const { return m_lastArgElementType; }
    ov::PartialShape get_lastArgPartialShape() const { return m_lastArgPartialShape; }
};

/// \brief Mock implementation of InputModel
/// Every call increments appropriate counters in statistic and stores argument values to statistics
/// as well
class MOCK_API InputModelMockPy : public InputModel
{
    static ModelStat m_stat;
    static ov::PartialShape m_returnShape;

    std::set<std::string> m_operations = {
        "8", "9", "8:9", "operation", "operation:0", "0:operation", "tensorAndOp", "conv2d"};
    std::set<std::string> m_tensors = {"8:9",
                                       "tensor",
                                       "tensor:0",
                                       "0:tensor",
                                       "tensorAndOp:0",
                                       "conv2d:0",
                                       "0:conv2d",
                                       "mock_input1",
                                       "mock_input2",
                                       "newInput1",
                                       "newIn1",
                                       "newIn2",
                                       "mock_output1",
                                       "mock_output2",
                                       "new_output2",
                                       "newOut1",
                                       "newOut2"};

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

    Place::Ptr get_place_by_operation_name(const std::string& opName) const override
    {
        m_stat.m_get_place_by_operation_name++;
        m_stat.m_lastArgString = opName;
        if (m_operations.count(opName))
        {
            return std::make_shared<PlaceMockPy>(opName, true);
        }
        return nullptr;
    }

    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override
    {
        m_stat.m_get_place_by_tensor_name++;
        m_stat.m_lastArgString = tensorName;
        if (m_tensors.count(tensorName))
        {
            return std::make_shared<PlaceMockPy>(tensorName);
        }
        return nullptr;
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
    void set_partial_shape(const Place::Ptr& place, const ov::PartialShape& shape) override
    {
        m_stat.m_set_partial_shape++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgPartialShape = shape;
    }

    ov::PartialShape get_partial_shape(const Place::Ptr& place) const override
    {
        m_stat.m_get_partial_shape++;
        m_stat.m_lastArgPlace = place;
        return m_returnShape;
    }

    void set_element_type(const Place::Ptr& place, const ov::element::Type& type) override
    {
        m_stat.m_set_element_type++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgElementType = type;
    }

    static void mock_return_partial_shape(const ov::PartialShape& shape) { m_returnShape = shape; }

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
    int m_supported = 0;
    int m_get_name = 0;
    // Getters
    std::vector<std::string> load_paths() const { return m_load_paths; }
    int convert_model() const { return m_convert_model; }
    int supported() const { return m_supported; }
    int get_name() const { return m_get_name; }
};

/// \brief Mock implementation of FrontEnd
/// Every call increments appropriate counters in statistic and stores argument values to statistics
/// as well
class MOCK_API FrontEndMockPy : public FrontEnd
{
    static FeStat m_stat;

public:
    FrontEndMockPy() = default;

    std::shared_ptr<ov::Model> convert(const InputModel::Ptr& model) const override
    {
        std::cout << "MVN: convert called\n";
        m_stat.m_convert_model++;
        return std::make_shared<ov::Model>(ov::NodeVector{}, ov::ParameterVector{});
    }

    static FeStat get_stat() { return m_stat; }

    static void clear_stat() { m_stat = {}; }

private:
    InputModel::Ptr load_impl(const std::vector<ov::Any>& params) const override
    {
        if (!params.empty() && params[0].is<std::string>())
        {
            auto path = params[0].as<std::string>();
            m_stat.m_load_paths.push_back(path);
        }
        return std::make_shared<InputModelMockPy>();
    }

    bool supported_impl(const std::vector<ov::Any>& params) const override
    {
        m_stat.m_supported++;
        if (!params.empty() && params[0].is<std::string>())
        {
            auto path = params[0].as<std::string>();
            if (path.find(".test_mo_mock_mdl") != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    std::string get_name() const override
    {
        m_stat.m_get_name++;
        return "openvino_mock_mo_frontend";
    }
};
