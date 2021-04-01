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

#include "ngraph/function.hpp"     // ngraph::Function
#include "ngraph/op/parameter.hpp" // ngraph::op::Parameter
#include "pyngraph/function.hpp"

namespace py = pybind11;

static const char* CAPSULE_NAME = "ngraph_function";

void regclass_pyngraph_Function(py::module m)
{
    py::class_<ngraph::Function, std::shared_ptr<ngraph::Function>> function(m, "Function");
    function.doc() = "ngraph.impl.Function wraps ngraph::Function";
    function.def(py::init<const std::vector<std::shared_ptr<ngraph::Node>>&,
                          const std::vector<std::shared_ptr<ngraph::op::Parameter>>&,
                          const std::string&>());
    function.def(py::init<const std::shared_ptr<ngraph::Node>&,
                          const std::vector<std::shared_ptr<ngraph::op::Parameter>>&,
                          const std::string&>());
    function.def("get_output_size", &ngraph::Function::get_output_size);
    function.def("get_ops", &ngraph::Function::get_ops);
    function.def("get_ordered_ops", &ngraph::Function::get_ordered_ops);
    function.def("get_output_op", &ngraph::Function::get_output_op);
    function.def("get_output_element_type", &ngraph::Function::get_output_element_type);
    function.def("get_output_shape", &ngraph::Function::get_output_shape);
    function.def("get_output_partial_shape", &ngraph::Function::get_output_partial_shape);
    function.def("get_parameters", &ngraph::Function::get_parameters);
    function.def("get_results", &ngraph::Function::get_results);
    function.def("get_result", &ngraph::Function::get_result);
    function.def("get_name", &ngraph::Function::get_name);
    function.def("get_friendly_name", &ngraph::Function::get_friendly_name);
    function.def("set_friendly_name", &ngraph::Function::set_friendly_name);
    function.def("is_dynamic", &ngraph::Function::is_dynamic);
    function.def("__repr__", [](const ngraph::Function& self) {
        std::string class_name = py::cast(self).get_type().attr("__name__").cast<std::string>();
        std::stringstream shapes_ss;
        for (size_t i = 0; i < self.get_output_size(); ++i)
        {
            if (i > 0)
            {
                shapes_ss << ", ";
            }
            shapes_ss << self.get_output_partial_shape(i);
        }
        return "<" + class_name + ": '" + self.get_friendly_name() + "' (" + shapes_ss.str() + ")>";
    });
    function.def_static("from_capsule", [](py::object* capsule) {
        // get the underlying PyObject* which is a PyCapsule pointer
        auto* pybind_capsule_ptr = capsule->ptr();
        // extract the pointer stored in the PyCapsule under the name CAPSULE_NAME
        auto* capsule_ptr = PyCapsule_GetPointer(pybind_capsule_ptr, CAPSULE_NAME);

        auto* ngraph_function = static_cast<std::shared_ptr<ngraph::Function>*>(capsule_ptr);
        if (ngraph_function && *ngraph_function)
        {
            return *ngraph_function;
        }
        else
        {
            throw std::runtime_error("The provided capsule does not contain an ngraph::Function");
        }
    });
    function.def_static("to_capsule", [](std::shared_ptr<ngraph::Function>& ngraph_function) {
        // create a shared pointer on the heap before putting it in the capsule
        // this secures the lifetime of the object transferred by the capsule
        auto* sp_copy = new std::shared_ptr<ngraph::Function>(ngraph_function);

        // a destructor callback that will delete the heap allocated shared_ptr
        // when the capsule is destructed
        auto sp_deleter = [](PyObject* capsule) {
            auto* capsule_ptr = PyCapsule_GetPointer(capsule, CAPSULE_NAME);
            auto* function_sp = static_cast<std::shared_ptr<ngraph::Function>*>(capsule_ptr);
            if (function_sp)
            {
                delete function_sp;
            }
        };

        // put the shared_ptr in a new capsule under the same name as in "from_capsule"
        auto pybind_capsule = py::capsule(sp_copy, CAPSULE_NAME, sp_deleter);

        return pybind_capsule;
    });

    function.def_property_readonly("name", &ngraph::Function::get_name);
    function.def_property("friendly_name",
                          &ngraph::Function::get_friendly_name,
                          &ngraph::Function::set_friendly_name);
}
