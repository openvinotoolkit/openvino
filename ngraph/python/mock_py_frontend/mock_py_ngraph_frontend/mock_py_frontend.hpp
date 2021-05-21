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

class MOCK_API PlaceMockPy : public Place
{
};

////////////////////////////////

struct MOCK_API MdlCallStat
{
    FrontEndCapFlags m_loadFlags;
    std::vector<std::string> m_loadPaths;
    int m_getInputsCount = 0;
    int m_getOutputsCount = 0;
    int m_getPlaceByTensorNameCount = 0;
    int m_getPlaceByOperationNameCount = 0;
    int m_getPlaceByOperationNameAndInputPortCount = 0;
    int m_getPlaceByOperationNameAndOutputPortCount = 0;
    int m_setNameForTensorCount = 0;
    int m_addNameForTensorCount = 0;
    int m_setNameForOperationCount = 0;
    int m_freeNameForTensorCount = 0;
    int m_freeNameForOperationCount = 0;
    int m_setNameForDimensionCount = 0;
    int m_cutAndAddNewInputCount = 0;
    int m_cutAndAddNewOutputCount = 0;
    int m_addOutputCount = 0;
    int m_removeOutputCount = 0;
    int m_setPartialShapeCount = 0;
    int m_getPartialShapeCount = 0;
    int m_setElementTypeCount = 0;

    int m_extractSubgraphCount = 0;
    int m_overrideAllInputsCount = 0;
    int m_overrideAllOutputsCount = 0;

    // Arguments tracking
    std::string m_lastArgString;
    int m_lastArgInt;
    Place::Ptr m_lastArgPlace = nullptr;
    std::vector<Place::Ptr> m_lastArgInputPlaces;
    std::vector<Place::Ptr> m_lastArgOutputPlaces;
    ngraph::element::Type m_lastArgElementType;
    ngraph::PartialShape m_lastArgPartialShape;

    // Getters
    int get_getInputsCount() const { return m_getInputsCount; }
    int get_getOutputsCount() const { return m_getOutputsCount; }
    int get_extractSubgraphCount() const { return m_extractSubgraphCount; }
    int get_overrideAllInputsCount() const { return m_overrideAllInputsCount; }
    int get_overrideAllOutputsCount() const { return m_overrideAllOutputsCount; }
    int get_getPlaceByTensorNameCount() const { return m_getPlaceByTensorNameCount; }
    int get_getPlaceByOperationNameCount() const { return m_getPlaceByOperationNameCount; }
    int get_getPlaceByOperationNameAndInputPortCount() const
    {
        return m_getPlaceByOperationNameAndInputPortCount;
    }
    int get_getPlaceByOperationNameAndOutputPortCount() const
    {
        return m_getPlaceByOperationNameAndOutputPortCount;
    }
    int get_setNameForTensorCount() const { return m_setNameForTensorCount; }
    int get_addNameForTensorCount() const { return m_addNameForTensorCount; }
    int get_setNameForOperationCount() const { return m_setNameForOperationCount; }
    int get_freeNameForTensorCount() const { return m_freeNameForTensorCount; }
    int get_freeNameForOperationCount() const { return m_freeNameForOperationCount; }
    int get_setNameForDimensionCount() const { return m_setNameForDimensionCount; }
    int get_cutAndAddNewInputCount() const { return m_cutAndAddNewInputCount; }
    int get_cutAndAddNewOutputCount() const { return m_cutAndAddNewOutputCount; }
    int get_addOutputCount() const { return m_addOutputCount; }
    int get_removeOutputCount() const { return m_removeOutputCount; }
    int get_setPartialShapeCount() const { return m_setPartialShapeCount; }
    int get_getPartialShapeCount() const { return m_getPartialShapeCount; }
    int get_setElementTypeCount() const { return m_setElementTypeCount; }

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
    mutable MdlCallStat m_stat;

public:
    std::vector<Place::Ptr> get_inputs() const override
    {
        m_stat.m_getInputsCount++;
        return {std::make_shared<PlaceMockPy>()};
    }

    std::vector<Place::Ptr> get_outputs() const override
    {
        m_stat.m_getOutputsCount++;
        return {std::make_shared<PlaceMockPy>()};
    }

    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override
    {
        m_stat.m_getPlaceByTensorNameCount++;
        m_stat.m_lastArgString = tensorName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_place_by_operation_name(const std::string& operationName) override
    {
        m_stat.m_getPlaceByOperationNameCount++;
        m_stat.m_lastArgString = operationName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_place_by_operation_and_input_port(const std::string& operationName,
                                                     int inputPortIndex) override
    {
        m_stat.m_getPlaceByOperationNameAndInputPortCount++;
        m_stat.m_lastArgInt = inputPortIndex;
        m_stat.m_lastArgString = operationName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_place_by_operation_and_output_port(const std::string& operationName,
                                                      int outputPortIndex) override
    {
        m_stat.m_getPlaceByOperationNameAndOutputPortCount++;
        m_stat.m_lastArgInt = outputPortIndex;
        m_stat.m_lastArgString = operationName;
        return std::make_shared<PlaceMockPy>();
    }

    void set_name_for_tensor(Place::Ptr tensor, const std::string& newName) override
    {
        m_stat.m_setNameForTensorCount++;
        m_stat.m_lastArgPlace = tensor;
        m_stat.m_lastArgString = newName;
    }

    void add_name_for_tensor(Place::Ptr tensor, const std::string& newName) override
    {
        m_stat.m_addNameForTensorCount++;
        m_stat.m_lastArgPlace = tensor;
        m_stat.m_lastArgString = newName;
    }

    void set_name_for_operation(Place::Ptr operation, const std::string& newName) override
    {
        m_stat.m_setNameForOperationCount++;
        m_stat.m_lastArgPlace = operation;
        m_stat.m_lastArgString = newName;
    }

    void free_name_for_tensor(const std::string& name) override
    {
        m_stat.m_freeNameForTensorCount++;
        m_stat.m_lastArgString = name;
    }

    void free_name_for_operation(const std::string& name) override
    {
        m_stat.m_freeNameForOperationCount++;
        m_stat.m_lastArgString = name;
    }

    void set_name_for_dimension(Place::Ptr place,
                                size_t shapeDimIndex,
                                const std::string& dimName) override
    {
        m_stat.m_setNameForDimensionCount++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgInt = static_cast<int>(shapeDimIndex);
        m_stat.m_lastArgString = dimName;
    }

    void cut_and_add_new_input(Place::Ptr place, const std::string& newNameOptional) override
    {
        m_stat.m_cutAndAddNewInputCount++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgString = newNameOptional;
    }

    void cut_and_add_new_output(Place::Ptr place, const std::string& newNameOptional) override
    {
        m_stat.m_cutAndAddNewOutputCount++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgString = newNameOptional;
    }

    Place::Ptr add_output(Place::Ptr place) override
    {
        m_stat.m_addOutputCount++;
        m_stat.m_lastArgPlace = place;
        return std::make_shared<PlaceMockPy>();
    }

    void remove_output(Place::Ptr place) override
    {
        m_stat.m_removeOutputCount++;
        m_stat.m_lastArgPlace = place;
    }

    void override_all_outputs(const std::vector<Place::Ptr>& outputs) override
    {
        m_stat.m_overrideAllOutputsCount++;
        m_stat.m_lastArgOutputPlaces = outputs;
    }

    void override_all_inputs(const std::vector<Place::Ptr>& inputs) override
    {
        m_stat.m_overrideAllInputsCount++;
        m_stat.m_lastArgInputPlaces = inputs;
    }

    void extract_subgraph(const std::vector<Place::Ptr>& inputs,
                          const std::vector<Place::Ptr>& outputs) override
    {
        m_stat.m_extractSubgraphCount++;
        m_stat.m_lastArgInputPlaces = inputs;
        m_stat.m_lastArgOutputPlaces = outputs;
    }

    // Setting tensor properties
    void set_partial_shape(Place::Ptr place, const ngraph::PartialShape& shape) override
    {
        m_stat.m_setPartialShapeCount++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgPartialShape = shape;
    }

    ngraph::PartialShape get_partial_shape(Place::Ptr place) const override
    {
        m_stat.m_getPartialShapeCount++;
        m_stat.m_lastArgPlace = place;
        return {};
    }

    void set_element_type(Place::Ptr place, const ngraph::element::Type& type) override
    {
        m_stat.m_setElementTypeCount++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgElementType = type;
    }

    //---------------Stat--------------------
    MdlCallStat get_stat() const { return m_stat; }

    void reset_stat() { m_stat = {}; }
};

/////////////////////////////////////////////////////////

struct MOCK_API FeCallStat
{
    FrontEndCapFlags m_loadFlags;
    std::vector<std::string> m_loadPaths;
    int m_convertModelCount = 0;
    int m_convertFuncCount = 0;
    int m_convertPartCount = 0;
    int m_decodeCount = 0;
    int m_normalizeCount = 0;
    // Getters
    FrontEndCapFlags get_loadFlags() const { return m_loadFlags; }
    std::vector<std::string> get_loadPaths() const { return m_loadPaths; }
    int get_convertModelCount() const { return m_convertModelCount; }
    int get_convertFuncCount() const { return m_convertFuncCount; }
    int get_convertPartCount() const { return m_convertPartCount; }
    int get_decodeCount() const { return m_decodeCount; }
    int get_normalizeCount() const { return m_normalizeCount; }
};

class MOCK_API FrontEndMockPy : public FrontEnd
{
    mutable FeCallStat m_stat;

public:
    FrontEndMockPy(FrontEndCapFlags flags) { m_stat.m_loadFlags = flags; }

    InputModel::Ptr load_from_file(const std::string& path) const override
    {
        m_stat.m_loadPaths.push_back(path);
        return std::make_shared<InputModelMockPy>();
    }

    std::shared_ptr<ngraph::Function> convert(InputModel::Ptr model) const override
    {
        m_stat.m_convertModelCount++;
        return std::make_shared<ngraph::Function>(NodeVector{}, ParameterVector{});
    }

    std::shared_ptr<ngraph::Function> convert(std::shared_ptr<ngraph::Function> func) const override
    {
        m_stat.m_convertFuncCount++;
        return func;
    }

    std::shared_ptr<ngraph::Function> convert_partially(InputModel::Ptr model) const override
    {
        m_stat.m_convertPartCount++;
        return std::make_shared<ngraph::Function>(NodeVector{}, ParameterVector{});
    }

    std::shared_ptr<ngraph::Function> decode(InputModel::Ptr model) const override
    {
        m_stat.m_decodeCount++;
        return std::make_shared<ngraph::Function>(NodeVector{}, ParameterVector{});
    }

    void normalize(std::shared_ptr<ngraph::Function> function) const override
    {
        m_stat.m_normalizeCount++;
    }

    FeCallStat get_stat() const { return m_stat; }

    void reset_stat() { m_stat = {}; }
};
