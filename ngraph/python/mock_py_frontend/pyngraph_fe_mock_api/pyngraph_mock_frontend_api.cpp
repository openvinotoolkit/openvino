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
    mdlStat.def_property_readonly("lastPlaceName", &MdlCallStat::get_lastPlaceName);
    mdlStat.def_property_readonly("lastPlacePortIndex", &MdlCallStat::get_lastPlacePortIndex);
}

PYBIND11_MODULE(pybind_mock_frontend, m)
{
    m.doc() = "Mock frontend call counters for testing Pyngraph frontend bindings";
    register_mock_frontend_stat(m);
    register_mock_model_stat(m);
}
