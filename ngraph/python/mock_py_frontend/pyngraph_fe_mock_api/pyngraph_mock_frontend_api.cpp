// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../mock_py_ngraph_frontend/mock_py_frontend.hpp"

namespace py = pybind11;
using namespace ngraph;
using namespace ngraph::frontend;

static void register_mock_frontend_stat(py::module m)
{
    m.def(
        "get_fe_stat",
        [](const std::shared_ptr<FrontEnd>& fe) {
            std::shared_ptr<FrontEndMockPy> ptr = std::dynamic_pointer_cast<FrontEndMockPy>(fe);
            if (ptr)
            {
                auto stat = ptr->get_stat();
                return stat;
            }
            return FeCallStat();
        },
        py::arg("frontend"));

    m.def(
        "reset_fe_stat",
        [](const FrontEnd::Ptr& fe) {
            std::shared_ptr<FrontEndMockPy> ptr = std::dynamic_pointer_cast<FrontEndMockPy>(fe);
            if (ptr)
            {
                ptr->reset_stat();
            }
        },
        py::arg("frontend"));

    py::class_<FeCallStat> feStat(m, "FeStat", py::dynamic_attr());
    feStat.def_property_readonly("load_flags", &FeCallStat::get_loadFlags);
    feStat.def_property_readonly("loaded_paths", &FeCallStat::get_loadPaths);
    feStat.def_property_readonly("convertModelCount", &FeCallStat::get_convertModelCount);
    feStat.def_property_readonly("convertFuncCount", &FeCallStat::get_convertFuncCount);
    feStat.def_property_readonly("convertPartCount", &FeCallStat::get_convertPartCount);
    feStat.def_property_readonly("decodeCount", &FeCallStat::get_decodeCount);
    feStat.def_property_readonly("normalizeCount", &FeCallStat::get_normalizeCount);
}

static void register_mock_model_stat(py::module m)
{
    m.def(
        "get_mdl_stat",
        [](const std::shared_ptr<InputModel>& mdl) {
            std::shared_ptr<InputModelMockPy> ptr =
                std::dynamic_pointer_cast<InputModelMockPy>(mdl);
            if (ptr)
            {
                auto stat = ptr->get_stat();
                return stat;
            }
            return MdlCallStat();
        },
        py::arg("model"));

    m.def(
        "reset_mdl_stat",
        [](const std::shared_ptr<InputModel>& fe) {
            std::shared_ptr<InputModelMockPy> ptr = std::dynamic_pointer_cast<InputModelMockPy>(fe);
            if (ptr)
            {
                ptr->reset_stat();
            }
        },
        py::arg("model"));

    py::class_<MdlCallStat> mdlStat(m, "ModelStat", py::dynamic_attr());
    mdlStat.def_property_readonly("getInputsCount", &MdlCallStat::get_getInputsCount);
    mdlStat.def_property_readonly("getOutputsCount", &MdlCallStat::get_getOutputsCount);
    mdlStat.def_property_readonly("getPlaceByTensorNameCount",
                                  &MdlCallStat::get_getPlaceByTensorNameCount);
    mdlStat.def_property_readonly("getPlaceByOperationNameCount",
                                  &MdlCallStat::get_getPlaceByOperationNameCount);
    mdlStat.def_property_readonly("getPlaceByOperationNameAndInputPortCount",
                                  &MdlCallStat::get_getPlaceByOperationNameAndInputPortCount);
    mdlStat.def_property_readonly("getPlaceByOperationNameAndOutputPortCount",
                                  &MdlCallStat::get_getPlaceByOperationNameAndOutputPortCount);

    mdlStat.def_property_readonly("setNameForTensorCount", &MdlCallStat::get_setNameForTensorCount);
    mdlStat.def_property_readonly("addNameForTensorCount", &MdlCallStat::get_addNameForTensorCount);
    mdlStat.def_property_readonly("setNameForOperationCount",
                                  &MdlCallStat::get_setNameForOperationCount);
    mdlStat.def_property_readonly("freeNameForTensorCount",
                                  &MdlCallStat::get_freeNameForTensorCount);
    mdlStat.def_property_readonly("freeNameForOperationCount",
                                  &MdlCallStat::get_freeNameForOperationCount);
    mdlStat.def_property_readonly("setNameForDimensionCount",
                                  &MdlCallStat::get_setNameForDimensionCount);
    mdlStat.def_property_readonly("cutAndAddNewInputCount",
                                  &MdlCallStat::get_cutAndAddNewInputCount);
    mdlStat.def_property_readonly("cutAndAddNewOutputCount",
                                  &MdlCallStat::get_cutAndAddNewOutputCount);
    mdlStat.def_property_readonly("addOutputCount", &MdlCallStat::get_addOutputCount);
    mdlStat.def_property_readonly("removeOutputCount", &MdlCallStat::get_removeOutputCount);
    mdlStat.def_property_readonly("setPartialShapeCount", &MdlCallStat::get_setPartialShapeCount);
    mdlStat.def_property_readonly("getPartialShapeCount", &MdlCallStat::get_getPartialShapeCount);
    mdlStat.def_property_readonly("setElementTypeCount", &MdlCallStat::get_setElementTypeCount);
    mdlStat.def_property_readonly("extractSubgraphCount", &MdlCallStat::get_extractSubgraphCount);
    mdlStat.def_property_readonly("overrideAllInputsCount",
                                  &MdlCallStat::get_overrideAllInputsCount);
    mdlStat.def_property_readonly("overrideAllOutputsCount",
                                  &MdlCallStat::get_overrideAllOutputsCount);

    // Arguments tracking
    mdlStat.def_property_readonly("lastArgString", &MdlCallStat::get_lastArgString);
    mdlStat.def_property_readonly("lastArgInt", &MdlCallStat::get_lastArgInt);
    mdlStat.def_property_readonly("lastArgPlace", &MdlCallStat::get_lastArgPlace);
    mdlStat.def_property_readonly("lastArgInputPlaces", &MdlCallStat::get_lastArgInputPlaces);
    mdlStat.def_property_readonly("lastArgOutputPlaces", &MdlCallStat::get_lastArgOutputPlaces);
    mdlStat.def_property_readonly("lastArgElementType", &MdlCallStat::get_lastArgElementType);
    mdlStat.def_property_readonly("lastArgPartialShape", &MdlCallStat::get_lastArgPartialShape);
}

PYBIND11_MODULE(pybind_mock_frontend, m)
{
    m.doc() = "Mock frontend call counters for testing Pyngraph frontend bindings";
    register_mock_frontend_stat(m);
    register_mock_model_stat(m);
}
