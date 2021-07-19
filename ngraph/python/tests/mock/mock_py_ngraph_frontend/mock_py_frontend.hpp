// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "frontend_manager/frontend_manager.hpp"
#include "frontend_manager/frontend_manager_defs.hpp"
#include "ngraph/visibility.hpp"

// Defined if we are building the plugin DLL (instead of using it)
#ifdef mock_py_ngraph_frontend_EXPORTS
#define MOCK_API NGRAPH_HELPER_DLL_EXPORT
#else
#define MOCK_API NGRAPH_HELPER_DLL_IMPORT
#endif // mock1_ngraph_frontend_EXPORTS

// OK to have 'using' in mock header

using namespace ngraph;
using namespace ngraph::frontend;

////////////////////////////////

struct MOCK_API PlaceStat
{
    int m_get_names = 0;
    int m_get_consuming_operations = 0;
    int m_get_target_tensor = 0;
    int m_get_producing_operation = 0;
    int m_get_producing_port = 0;
    int m_get_input_port = 0;
    int m_get_output_port = 0;
    int m_get_consuming_ports = 0;
    int m_is_input = 0;
    int m_is_output = 0;
    int m_is_equal = 0;
    int m_is_equal_data = 0;
    int m_get_source_tensor = 0;

    // Arguments tracking
    std::string m_lastArgString;
    int m_lastArgInt;
    Place::Ptr m_lastArgPlace = nullptr;

    // Getters
    int get_names() const { return m_get_names; }
    int get_consuming_operations() const { return m_get_consuming_operations; }
    int get_target_tensor() const { return m_get_target_tensor; }
    int get_producing_operation() const { return m_get_producing_operation; }
    int get_producing_port() const { return m_get_producing_port; }
    int get_input_port() const { return m_get_input_port; }
    int get_output_port() const { return m_get_output_port; }
    int get_consuming_ports() const { return m_get_consuming_ports; }
    int is_input() const { return m_is_input; }
    int is_output() const { return m_is_output; }
    int is_equal() const { return m_is_equal; }
    int is_equal_data() const { return m_is_equal_data; }
    int get_source_tensor() const { return m_get_source_tensor; }

    // Arguments getters
    std::string get_lastArgString() const { return m_lastArgString; }
    int get_lastArgInt() const { return m_lastArgInt; }
    Place::Ptr get_lastArgPlace() const { return m_lastArgPlace; }
};

class MOCK_API PlaceMockPy : public Place
{
    mutable PlaceStat m_stat;

public:
    std::vector<std::string> get_names() const override
    {
        m_stat.m_get_names++;
        return {};
    }

    std::vector<Place::Ptr> get_consuming_operations() const override
    {
        m_stat.m_get_consuming_operations++;
        m_stat.m_lastArgInt = -1;
        return {std::make_shared<PlaceMockPy>()};
    }

    std::vector<Place::Ptr> get_consuming_operations(int outputPortIndex) const override
    {
        m_stat.m_get_consuming_operations++;
        m_stat.m_lastArgInt = outputPortIndex;
        return {std::make_shared<PlaceMockPy>()};
    }

    Place::Ptr get_target_tensor() const override
    {
        m_stat.m_get_target_tensor++;
        m_stat.m_lastArgInt = -1;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_target_tensor(int outputPortIndex) const override
    {
        m_stat.m_get_target_tensor++;
        m_stat.m_lastArgInt = outputPortIndex;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_producing_operation() const override
    {
        m_stat.m_get_producing_operation++;
        m_stat.m_lastArgInt = -1;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_producing_operation(int inputPortIndex) const override
    {
        m_stat.m_get_producing_operation++;
        m_stat.m_lastArgInt = inputPortIndex;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_producing_port() const override
    {
        m_stat.m_get_producing_port++;
        return std::make_shared<PlaceMockPy>();
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

    std::vector<Place::Ptr> get_consuming_ports() const override
    {
        m_stat.m_get_consuming_ports++;
        return {std::make_shared<PlaceMockPy>()};
    }

    bool is_input() const override
    {
        m_stat.m_is_input++;
        return false;
    }

    bool is_output() const override
    {
        m_stat.m_is_output++;
        return false;
    }

    bool is_equal(Ptr another) const override
    {
        m_stat.m_is_equal++;
        m_stat.m_lastArgPlace = another;
        return false;
    }

    bool is_equal_data(Ptr another) const override
    {
        m_stat.m_is_equal_data++;
        m_stat.m_lastArgPlace = another;
        return false;
    }

    Place::Ptr get_source_tensor(int inputPortIndex) const override
    {
        m_stat.m_get_source_tensor++;
        m_stat.m_lastArgInt = inputPortIndex;
        return {std::make_shared<PlaceMockPy>()};
    }

    Place::Ptr get_source_tensor() const override
    {
        m_stat.m_get_source_tensor++;
        m_stat.m_lastArgInt = -1;
        return {std::make_shared<PlaceMockPy>()};
    }

    //---------------Stat--------------------
    PlaceStat get_stat() const { return m_stat; }
};

////////////////////////////////

struct MOCK_API ModelStat
{
    int m_get_inputs = 0;
    int m_get_outputs = 0;
    int m_get_place_by_tensor_name = 0;
    int m_get_place_by_operation_name = 0;
    int m_get_place_by_operation_and_input_port = 0;
    int m_get_place_by_operation_and_output_port = 0;
    int m_set_name_for_tensor = 0;
    int m_add_name_for_tensor = 0;
    int m_set_name_for_operation = 0;
    int m_free_name_for_tensor = 0;
    int m_free_name_for_operation = 0;
    int m_set_name_for_dimension = 0;
    int m_cut_and_add_new_input = 0;
    int m_cut_and_add_new_output = 0;
    int m_add_output = 0;
    int m_remove_output = 0;
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
    int get_place_by_operation_name() const { return m_get_place_by_operation_name; }
    int get_place_by_operation_and_input_port() const
    {
        return m_get_place_by_operation_and_input_port;
    }
    int get_place_by_operation_and_output_port() const
    {
        return m_get_place_by_operation_and_output_port;
    }
    int set_name_for_tensor() const { return m_set_name_for_tensor; }
    int add_name_for_tensor() const { return m_add_name_for_tensor; }
    int set_name_for_operation() const { return m_set_name_for_operation; }
    int free_name_for_tensor() const { return m_free_name_for_tensor; }
    int free_name_for_operation() const { return m_free_name_for_operation; }
    int set_name_for_dimension() const { return m_set_name_for_dimension; }
    int cut_and_add_new_input() const { return m_cut_and_add_new_input; }
    int cut_and_add_new_output() const { return m_cut_and_add_new_output; }
    int add_output() const { return m_add_output; }
    int remove_output() const { return m_remove_output; }
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

class MOCK_API InputModelMockPy : public InputModel
{
    mutable ModelStat m_stat;

public:
    std::vector<Place::Ptr> get_inputs() const override
    {
        m_stat.m_get_inputs++;
        return {std::make_shared<PlaceMockPy>()};
    }

    std::vector<Place::Ptr> get_outputs() const override
    {
        m_stat.m_get_outputs++;
        return {std::make_shared<PlaceMockPy>()};
    }

    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override
    {
        m_stat.m_get_place_by_tensor_name++;
        m_stat.m_lastArgString = tensorName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_place_by_operation_name(const std::string& operationName) override
    {
        m_stat.m_get_place_by_operation_name++;
        m_stat.m_lastArgString = operationName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_place_by_operation_name_and_input_port(const std::string& operationName,
                                                          int inputPortIndex) override
    {
        m_stat.m_get_place_by_operation_and_input_port++;
        m_stat.m_lastArgInt = inputPortIndex;
        m_stat.m_lastArgString = operationName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_place_by_operation_name_and_output_port(const std::string& operationName,
                                                           int outputPortIndex) override
    {
        m_stat.m_get_place_by_operation_and_output_port++;
        m_stat.m_lastArgInt = outputPortIndex;
        m_stat.m_lastArgString = operationName;
        return std::make_shared<PlaceMockPy>();
    }

    void set_name_for_tensor(Place::Ptr tensor, const std::string& newName) override
    {
        m_stat.m_set_name_for_tensor++;
        m_stat.m_lastArgPlace = tensor;
        m_stat.m_lastArgString = newName;
    }

    void add_name_for_tensor(Place::Ptr tensor, const std::string& newName) override
    {
        m_stat.m_add_name_for_tensor++;
        m_stat.m_lastArgPlace = tensor;
        m_stat.m_lastArgString = newName;
    }

    void set_name_for_operation(Place::Ptr operation, const std::string& newName) override
    {
        m_stat.m_set_name_for_operation++;
        m_stat.m_lastArgPlace = operation;
        m_stat.m_lastArgString = newName;
    }

    void free_name_for_tensor(const std::string& name) override
    {
        m_stat.m_free_name_for_tensor++;
        m_stat.m_lastArgString = name;
    }

    void free_name_for_operation(const std::string& name) override
    {
        m_stat.m_free_name_for_operation++;
        m_stat.m_lastArgString = name;
    }

    void set_name_for_dimension(Place::Ptr place,
                                size_t shapeDimIndex,
                                const std::string& dimName) override
    {
        m_stat.m_set_name_for_dimension++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgInt = static_cast<int>(shapeDimIndex);
        m_stat.m_lastArgString = dimName;
    }

    void cut_and_add_new_input(Place::Ptr place, const std::string& newNameOptional) override
    {
        m_stat.m_cut_and_add_new_input++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgString = newNameOptional;
    }

    void cut_and_add_new_output(Place::Ptr place, const std::string& newNameOptional) override
    {
        m_stat.m_cut_and_add_new_output++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgString = newNameOptional;
    }

    Place::Ptr add_output(Place::Ptr place) override
    {
        m_stat.m_add_output++;
        m_stat.m_lastArgPlace = place;
        return std::make_shared<PlaceMockPy>();
    }

    void remove_output(Place::Ptr place) override
    {
        m_stat.m_remove_output++;
        m_stat.m_lastArgPlace = place;
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
        return {};
    }

    void set_element_type(Place::Ptr place, const ngraph::element::Type& type) override
    {
        m_stat.m_set_element_type++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgElementType = type;
    }

    //---------------Stat--------------------
    ModelStat get_stat() const { return m_stat; }
};

/////////////////////////////////////////////////////////

struct MOCK_API FeStat
{
    std::vector<std::string> m_load_paths;
    int m_convert_model = 0;
    int m_convert = 0;
    int m_convert_partially = 0;
    int m_decode = 0;
    int m_normalize = 0;
    // Getters
    std::vector<std::string> load_paths() const { return m_load_paths; }
    int convert_model() const { return m_convert_model; }
    int convert() const { return m_convert; }
    int convert_partially() const { return m_convert_partially; }
    int decode() const { return m_decode; }
    int normalize() const { return m_normalize; }
};

class MOCK_API FrontEndMockPy : public FrontEnd
{
    mutable FeStat m_stat;

public:
    FrontEndMockPy() {}

    InputModel::Ptr load_impl(const std::vector<std::shared_ptr<Variant>>& params) const override
    {
        if (params.size() > 0 && is_type<VariantWrapper<std::string>>(params[0]))
            m_stat.m_load_paths.push_back(as_type_ptr<VariantWrapper<std::string>>(params[0])->get());
        return std::make_shared<InputModelMockPy>();
    }

    std::shared_ptr<ngraph::Function> convert(InputModel::Ptr model) const override
    {
        m_stat.m_convert_model++;
        return std::make_shared<ngraph::Function>(NodeVector{}, ParameterVector{});
    }

    std::shared_ptr<ngraph::Function> convert(std::shared_ptr<ngraph::Function> func) const override
    {
        m_stat.m_convert++;
        return func;
    }

    std::shared_ptr<ngraph::Function> convert_partially(InputModel::Ptr model) const override
    {
        m_stat.m_convert_partially++;
        return std::make_shared<ngraph::Function>(NodeVector{}, ParameterVector{});
    }

    std::shared_ptr<ngraph::Function> decode(InputModel::Ptr model) const override
    {
        m_stat.m_decode++;
        return std::make_shared<ngraph::Function>(NodeVector{}, ParameterVector{});
    }

    void normalize(std::shared_ptr<ngraph::Function> function) const override
    {
        m_stat.m_normalize++;
    }

    FeStat get_stat() const { return m_stat; }
};
