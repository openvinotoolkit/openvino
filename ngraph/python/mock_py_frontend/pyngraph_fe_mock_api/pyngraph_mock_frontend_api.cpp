// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../mock_py_ngraph_frontend/mock_py_frontend.hpp"

namespace py = pybind11;
using namespace ngraph;
using namespace ngraph::frontend;

PYBIND11_MODULE(pybind_mock_frontend, m)
{
    m.doc() = "Mock frontend for testing Pyngraph frontend bindings";
    m.def("get_stat", [](const std::shared_ptr<FrontEnd>& fe) {
        std::shared_ptr<FrontEndMockPy> ptr = std::dynamic_pointer_cast<FrontEndMockPy>(fe);
        if (ptr) {
            auto stat = ptr->get_stat();
            return stat;
        }
        return FeCallStat();
    }, py::arg("frontend"));

    m.def("reset_stat", [](const std::shared_ptr<FrontEnd>& fe) {
        std::shared_ptr<FrontEndMockPy> ptr = std::dynamic_pointer_cast<FrontEndMockPy>(fe);
        if (ptr) {
            ptr->reset_stat();
        }
    }, py::arg("frontend"));

    py::class_<FeCallStat> feStat(m, "FeStat", py::dynamic_attr());
    feStat.def_property_readonly("load_flags", &FeCallStat::get_loadFlags);
    feStat.def_property_readonly("loaded_paths", &FeCallStat::get_loadPaths);
    feStat.def_property_readonly("convertModelCount", &FeCallStat::get_convertModelCount);
    feStat.def_property_readonly("convertFuncCount", &FeCallStat::get_convertFuncCount);
    feStat.def_property_readonly("convertPartCount", &FeCallStat::get_convertPartCount);
    feStat.def_property_readonly("decodeCount", &FeCallStat::get_decodeCount);
    feStat.def_property_readonly("normalizeCount", &FeCallStat::get_normalizeCount);
}
