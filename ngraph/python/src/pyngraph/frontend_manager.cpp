//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "frontend_manager/frontend_manager.hpp"
#include "pyngraph/function.hpp"

namespace py = pybind11;

void regclass_pyngraph_FrontEndManager(py::module m)
{
    py::class_<ngraph::frontend::FrontEndManager,
               std::shared_ptr<ngraph::frontend::FrontEndManager>>
        fem(m, "FrontEndManager", py::dynamic_attr());
    fem.doc() = "ngraph.impl.FrontEndManager wraps ngraph::frontend::FrontEndManager";

    fem.def(py::init<>());

    fem.def("get_available_front_ends",
            &ngraph::frontend::FrontEndManager::get_available_front_ends);
    fem.def("load_by_framework",
            &ngraph::frontend::FrontEndManager::load_by_framework,
            py::arg("framework"),
            py::arg("capabilities") = ngraph::frontend::FrontEndCapabilities::FEC_DEFAULT);
}

void regclass_pyngraph_FrontEnd(py::module m)
{
    py::class_<ngraph::frontend::FrontEnd, std::shared_ptr<ngraph::frontend::FrontEnd>> fem(
        m, "FrontEnd", py::dynamic_attr());
    fem.doc() = "ngraph.impl.FrontEnd wraps ngraph::frontend::FrontEnd";

    fem.def("load_from_file", &ngraph::frontend::FrontEnd::load_from_file, py::arg("path"));
    fem.def("convert",
            static_cast<std::shared_ptr<ngraph::Function> (ngraph::frontend::FrontEnd::*)(
                ngraph::frontend::InputModel::Ptr) const>(&ngraph::frontend::FrontEnd::convert));
    fem.def("convert",
            static_cast<std::shared_ptr<ngraph::Function> (ngraph::frontend::FrontEnd::*)(
                std::shared_ptr<ngraph::Function>) const>(&ngraph::frontend::FrontEnd::convert));
}

void regclass_pyngraph_Place(py::module m)
{
    py::class_<ngraph::frontend::Place, std::shared_ptr<ngraph::frontend::Place>> place(
        m, "Place", py::dynamic_attr());
    place.doc() = "ngraph.impl.Place wraps ngraph::frontend::Place";

    place.def("is_input", &ngraph::frontend::Place::is_input);
    place.def("is_output", &ngraph::frontend::Place::is_output);
    place.def("get_names", &ngraph::frontend::Place::get_names);
    place.def("is_equal", &ngraph::frontend::Place::is_equal);
}

void regclass_pyngraph_InputModel(py::module m)
{
    py::class_<ngraph::frontend::InputModel, std::shared_ptr<ngraph::frontend::InputModel>> im(
        m, "InputModel", py::dynamic_attr());
    im.doc() = "ngraph.impl.InputModel wraps ngraph::frontend::InputModel";
    im.def("extract_subgraph", &ngraph::frontend::InputModel::extract_subgraph);
    im.def("get_place_by_tensor_name", &ngraph::frontend::InputModel::get_place_by_tensor_name);
    im.def("set_partial_shape", &ngraph::frontend::InputModel::set_partial_shape);
    im.def("get_partial_shape", &ngraph::frontend::InputModel::get_partial_shape);
    im.def("get_inputs", &ngraph::frontend::InputModel::get_inputs);
    im.def("get_outputs", &ngraph::frontend::InputModel::get_outputs);
    im.def("override_all_inputs", &ngraph::frontend::InputModel::override_all_inputs);
    im.def("override_all_outputs", &ngraph::frontend::InputModel::override_all_outputs);
    im.def("set_element_type", &ngraph::frontend::InputModel::set_element_type);
}

void regclass_pyngraph_FEC(py::module m)
{
    class FeCaps
    {
    public:
        int get_caps() const { return m_caps; }

    private:
        int m_caps;
    };

    py::class_<FeCaps, std::shared_ptr<FeCaps>> type(m, "FrontEndCapabilities");
    // type.doc() = "FrontEndCapabilities";
    type.attr("DEFAULT") = ngraph::frontend::FrontEndCapabilities::FEC_DEFAULT;
    type.attr("CUT") = ngraph::frontend::FrontEndCapabilities::FEC_CUT;
    type.attr("NAMES") = ngraph::frontend::FrontEndCapabilities::FEC_NAMES;
    type.attr("WILDCARDS") = ngraph::frontend::FrontEndCapabilities::FEC_WILDCARDS;

    type.def(
        "__eq__",
        [](const FeCaps& a, const FeCaps& b) { return a.get_caps() == b.get_caps(); },
        py::is_operator());
}
