// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/function.hpp"     // ngraph::Function
#include "ngraph/op/parameter.hpp" // ngraph::op::Parameter
#include "ngraph/op/sink.hpp"
#include "pyngraph/function.hpp"

namespace py = pybind11;

static const char* CAPSULE_NAME = "ngraph_function";

void regclass_pyngraph_Function(py::module m)
{
    py::class_<ngraph::Function, std::shared_ptr<ngraph::Function>> function(m, "Function");
    function.doc() = "ngraph.impl.Function wraps ngraph::Function";

    function.def(py::init([](const ngraph::ResultVector& res,
                             const std::vector<std::shared_ptr<ngraph::Node>>& nodes,
                             const ngraph::ParameterVector& params,
                             const std::string& name) {
                     ngraph::SinkVector sinks;
                     for (const auto& node : nodes)
                     {
                         auto sink = std::dynamic_pointer_cast<ngraph::op::Sink>(node);
                         NGRAPH_CHECK(sink != nullptr, "Node {} is not instance of Sink");
                         sinks.push_back(sink);
                     }
                     return std::make_shared<ngraph::Function>(res, sinks, params, name);
                 }),
                 py::arg("results"),
                 py::arg("sinks"),
                 py::arg("parameters"),
                 py::arg("name"),
                 R"(
                    Create user-defined Function which is a representation of a model.

                    Parameters
                    ----------
                    results : List[op.Result]
                        List of results.

                    sinks : List[Node]
                        List of Nodes to be used as Sinks (e.g. Assign ops).

                    parameters : List[op.Parameter]
                        List of parameters.

                    name : str
                        String to set as function's friendly name.
                 )");

    function.def(py::init<const std::vector<std::shared_ptr<ngraph::Node>>&,
                          const std::vector<std::shared_ptr<ngraph::op::Parameter>>&,
                          const std::string&>(),
                 py::arg("results"),
                 py::arg("parameters"),
                 py::arg("name"),
                 R"(
                    Create user-defined Function which is a representation of a model.

                    Parameters
                    ----------
                    results : List[Node]
                        List of Nodes to be used as results.

                    parameters : List[op.Parameter]
                        List of parameters.

                    name : str
                        String to set as function's friendly name.
                 )");

    function.def(py::init<const std::shared_ptr<ngraph::Node>&,
                          const std::vector<std::shared_ptr<ngraph::op::Parameter>>&,
                          const std::string&>(),
                 py::arg("result"),
                 py::arg("parameters"),
                 py::arg("name"),
                 R"(
                    Create user-defined Function which is a representation of a model.

                    Parameters
                    ----------
                    results : Node
                        Node to be used as result.

                    parameters : List[op.Parameter]
                        List of parameters.

                    name : str
                        String to set as function's friendly name.
                 )");
    function.def("get_output_size",
                 &ngraph::Function::get_output_size,
                 R"(
                    Return the number of outputs for the function.

                    Returns
                    ----------
                    get_output_size : int
                        Number of outputs.
                 )");
    function.def("get_ops",
                 &ngraph::Function::get_ops,
                 R"(
                    Return ops used in the function.

                    Returns
                    ----------
                    get_ops : List[Node]
                        List of Nodes representing ops used in function.
                 )");
    function.def("get_ordered_ops",
                 &ngraph::Function::get_ordered_ops,
                 R"(
                    Return ops used in the function in topological order.

                    Returns
                    ----------
                    get_ordered_ops : List[Node]
                        List of sorted Nodes representing ops used in function.
                 )");
    function.def("get_output_op",
                 &ngraph::Function::get_output_op,
                 py::arg("i"),
                 R"(
                    Return the op that generates output i

                    Parameters
                    ----------
                    i : int
                        output index

                    Returns
                    ----------
                    get_output_op : Node
                        Node object that generates output i
                )");
    function.def("get_output_element_type",
                 &ngraph::Function::get_output_element_type,
                 py::arg("i"),
                 R"(
                    Return the element type of output i

                    Parameters
                    ----------
                    i : int
                        output index

                    Returns
                    ----------
                    get_output_op : Type
                        Type object of output i
                 )");
    function.def("get_output_shape",
                 &ngraph::Function::get_output_shape,
                 py::arg("i"),
                 R"(
                    Return the shape of element i

                    Parameters
                    ----------
                    i : int
                        element index

                    Returns
                    ----------
                    get_output_shape : Shape
                        Shape object of element i
                 )");
    function.def("get_output_partial_shape",
                 &ngraph::Function::get_output_partial_shape,
                 py::arg("i"),
                 R"(
                    Return the partial shape of element i

                    Parameters
                    ----------
                    i : int
                        element index

                    Returns
                    ----------
                    get_output_partial_shape : PartialShape
                        PartialShape object of element i
                 )");
    function.def("get_parameters",
                 &ngraph::Function::get_parameters,
                 R"(
                    Return the function parameters.

                    Returns
                    ----------
                    get_parameters : ParameterVector
                        ParameterVector containing function parameters.
                 )");
    function.def("get_results",
                 &ngraph::Function::get_results,
                 R"(
                    Return a list of function outputs.

                    Returns
                    ----------
                    get_results : ResultVector
                        ResultVector containing function parameters.
                 )");
    function.def("get_result",
                 &ngraph::Function::get_result,
                 R"(
                    Return single result.

                    Returns
                    ----------
                    get_result : Node
                        Node object representing result.
                 )");
    function.def("get_name",
                 &ngraph::Function::get_name,
                 R"(
                    Get the unique name of the function.

                    Returns
                    ----------
                    get_name : str
                        String with a name of the function.
                 )");
    function.def("get_friendly_name",
                 &ngraph::Function::get_friendly_name,
                 R"(
                    Gets the friendly name for a function. If no
                    friendly name has been set via set_friendly_name
                    then the function's unique name is returned.

                    Returns
                    ----------
                    get_friendly_name : str
                        String with a friendly name of the function.
                 )");
    function.def("set_friendly_name",
                 &ngraph::Function::set_friendly_name,
                 py::arg("name"),
                 R"(
                    Sets a friendly name for a function. This does
                    not overwrite the unique name of the function and
                    is retrieved via get_friendly_name(). Used mainly
                    for debugging.

                    Parameters
                    ----------
                    name : str
                        String to set as the friendly name.
                 )");
    function.def("is_dynamic",
                 &ngraph::Function::is_dynamic,
                 R"(
                    Returns true if any of the op's defined in the function 
                    contains partial shape.

                    Returns
                    ----------
                    is_dynamic : bool
                 )");
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
