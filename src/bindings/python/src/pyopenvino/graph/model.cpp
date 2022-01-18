// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/model.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/core/model.hpp"  // ov::Model
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/parameter.hpp"  // ov::op::v0::Parameter
#include "openvino/op/sink.hpp"
#include "pyopenvino/core/tensor.hpp"
#include "pyopenvino/graph/ops/result.hpp"
#include "pyopenvino/graph/ops/util/variable.hpp"
#include "pyopenvino/graph/rt_map.hpp"

namespace py = pybind11;

static const char* CAPSULE_NAME = "openvino_function";

using PyRTMap = ov::RTMap;

PYBIND11_MAKE_OPAQUE(PyRTMap);

void set_tensor_names(const ov::ParameterVector& parameters) {
    for (const auto& param : parameters) {
        ov::Output<ov::Node> p = param;
        if (p.get_node()->output(0).get_names().empty()) {
            std::unordered_set<std::string> p_names({p.get_node()->get_friendly_name()});
            p.get_node()->output(0).set_names(p_names);
        }
    }
}

ov::SinkVector cast_to_sink_vector(const std::vector<std::shared_ptr<ov::Node>>& nodes) {
    ov::SinkVector sinks;
    for (const auto& node : nodes) {
        auto sink = std::dynamic_pointer_cast<ov::op::Sink>(node);
        NGRAPH_CHECK(sink != nullptr, "Node {} is not instance of Sink");
        sinks.push_back(sink);
    }
    return sinks;
}

void regclass_graph_Model(py::module m) {
    py::class_<ov::Model, std::shared_ptr<ov::Model>> function(m, "Model", py::module_local());
    function.doc() = "openvino.runtime.Model wraps ov::Model";

    function.def(py::init([](const ov::ResultVector& res,
                             const std::vector<std::shared_ptr<ov::Node>>& nodes,
                             const ov::ParameterVector& params,
                             const std::string& name) {
                     set_tensor_names(params);
                     const auto sinks = cast_to_sink_vector(nodes);
                     return std::make_shared<ov::Model>(res, sinks, params, name);
                 }),
                 py::arg("results"),
                 py::arg("sinks"),
                 py::arg("parameters"),
                 py::arg("name"),
                 R"(
                    Create user-defined Model which is a representation of a model.

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

    function.def(py::init([](const std::vector<std::shared_ptr<ov::Node>>& results,
                             const ov::ParameterVector& parameters,
                             const std::string& name) {
                     set_tensor_names(parameters);
                     return std::make_shared<ov::Model>(results, parameters, name);
                 }),
                 py::arg("results"),
                 py::arg("parameters"),
                 py::arg("name") = "",
                 R"(
                    Create user-defined Model which is a representation of a model.

                    Parameters
                    ----------
                    results : List[Node]
                        List of Nodes to be used as results.

                    parameters : List[op.Parameter]
                        List of parameters.

                    name : str
                        String to set as function's friendly name.
                 )");

    function.def(py::init([](const std::shared_ptr<ov::Node>& result,
                             const ov::ParameterVector& parameters,
                             const std::string& name) {
                     set_tensor_names(parameters);
                     return std::make_shared<ov::Model>(result, parameters, name);
                 }),
                 py::arg("result"),
                 py::arg("parameters"),
                 py::arg("name") = "",
                 R"(
                    Create user-defined Model which is a representation of a model.

                    Parameters
                    ----------
                    result : Node
                        Node to be used as result.

                    parameters : List[op.Parameter]
                        List of parameters.

                    name : str
                        String to set as function's friendly name.
                 )");

    function.def(
        py::init([](const ov::OutputVector& results, const ov::ParameterVector& parameters, const std::string& name) {
            set_tensor_names(parameters);
            return std::make_shared<ov::Model>(results, parameters, name);
        }),
        py::arg("results"),
        py::arg("parameters"),
        py::arg("name") = "",
        R"(
            Create user-defined Model which is a representation of a model

            Parameters
            ----------
            results : List[Output]
                List of outputs.

            parameters : List[op.Parameter]
                List of parameters.

            name : str
                String to set as function's friendly name.
        )");

    function.def(py::init([](const ov::OutputVector& results,
                             const std::vector<std::shared_ptr<ov::Node>>& nodes,
                             const ov::ParameterVector& parameters,
                             const std::string& name) {
                     set_tensor_names(parameters);
                     const auto sinks = cast_to_sink_vector(nodes);
                     return std::make_shared<ov::Model>(results, sinks, parameters, name);
                 }),
                 py::arg("results"),
                 py::arg("sinks"),
                 py::arg("parameters"),
                 py::arg("name") = "",
                 R"(
            Create user-defined Model which is a representation of a model

            Parameters
            ----------
            results : List[Output]
                List of outputs.

            sinks : List[Node]
                List of Nodes to be used as Sinks (e.g. Assign ops).

            parameters : List[op.Parameter]
                List of parameters.

            name : str
                String to set as function's friendly name.
            )");
    function.def(py::init([](const ov::ResultVector& results,
                             const std::vector<std::shared_ptr<ov::Node>>& nodes,
                             const ov::ParameterVector& parameters,
                             const ov::op::util::VariableVector& variables,
                             const std::string& name) {
                     set_tensor_names(parameters);
                     const auto sinks = cast_to_sink_vector(nodes);
                     return std::make_shared<ov::Model>(results, sinks, parameters, variables, name);
                 }),
                 py::arg("results"),
                 py::arg("sinks"),
                 py::arg("parameters"),
                 py::arg("variables"),
                 py::arg("name") = "",
                 R"(
            Create user-defined Model which is a representation of a model

            Parameters
            ----------
            results : List[op.Result]
                List of results.

            sinks : List[Node]
                List of Nodes to be used as Sinks (e.g. Assign ops).

            parameters : List[op.Parameter]
                List of parameters.

            variables : List[op.util.Variable]
                List of variables.

            name : str
                String to set as function's friendly name.
            )");

    function.def(py::init([](const ov::OutputVector& results,
                             const std::vector<std::shared_ptr<ov::Node>>& nodes,
                             const ov::ParameterVector& parameters,
                             const ov::op::util::VariableVector& variables,
                             const std::string& name) {
                     set_tensor_names(parameters);
                     const auto sinks = cast_to_sink_vector(nodes);
                     return std::make_shared<ov::Model>(results, sinks, parameters, variables, name);
                 }),
                 py::arg("results"),
                 py::arg("sinks"),
                 py::arg("parameters"),
                 py::arg("variables"),
                 py::arg("name") = "",
                 R"(
            Create user-defined Model which is a representation of a model

            Parameters
            ----------
            results : List[Output]
                List of results.

            sinks : List[Node]
                List of Nodes to be used as Sinks (e.g. Assign ops).

            parameters : List[op.Parameter]
                List of parameters.

            variables : List[op.util.Variable]
                List of variables.

            name : str
                String to set as function's friendly name.
        )");

    function.def(py::init([](const ov::ResultVector& results,
                             const ov::ParameterVector& parameters,
                             const ov::op::util::VariableVector& variables,
                             const std::string& name) {
                     set_tensor_names(parameters);
                     return std::make_shared<ov::Model>(results, parameters, variables, name);
                 }),
                 py::arg("results"),
                 py::arg("parameters"),
                 py::arg("variables"),
                 py::arg("name") = "",
                 R"(
            Create user-defined Model which is a representation of a model

            Parameters
            ----------
            results : List[op.Result]
                List of results.

            parameters : List[op.Parameter]
                List of parameters.

            variables : List[op.util.Variable]
                List of variables.

            name : str
                String to set as function's friendly name.
        )");

    function.def(py::init([](const ov::OutputVector& results,
                             const ov::ParameterVector& parameters,
                             const ov::op::util::VariableVector& variables,
                             const std::string& name) {
                     set_tensor_names(parameters);
                     return std::make_shared<ov::Model>(results, parameters, variables, name);
                 }),
                 py::arg("results"),
                 py::arg("parameters"),
                 py::arg("variables"),
                 py::arg("name") = "",
                 R"(
            Create user-defined Model which is a representation of a model

            Parameters
            ----------
            results : List[Output]
                List of results.

            parameters : List[op.Parameter]
                List of parameters.

            variables : List[op.util.Variable]
                List of variables.

            name : str
                String to set as function's friendly name.
        )");

    function.def("validate_nodes_and_infer_types", &ov::Model::validate_nodes_and_infer_types);

    function.def(
        "reshape",
        [](ov::Model& self, const ov::PartialShape& partial_shape) {
            self.reshape(partial_shape);
        },
        py::arg("partial_shapes"),
        R"(
                Parameters
                ----------
                partial_shapes : PartialShape
                    Index of Output.

                Returns
                ----------
                reshape : void
             )");

    function.def(
        "reshape",
        [](ov::Model& self, const std::map<size_t, ov::PartialShape>& partial_shapes) {
            self.reshape(partial_shapes);
        },
        py::arg("partial_shapes"),
        R"(
                Parameters
                ----------
                partial_shapes : Dict[int, PartialShape]
                    Index of Output.

                Returns
                ----------
                reshape : void
             )");

    function.def(
        "reshape",
        [](ov::Model& self, const std::map<std::string, ov::PartialShape>& partial_shapes) {
            self.reshape(partial_shapes);
        },
        py::arg("partial_shapes"),
        R"(
                Parameters
                ----------
                partial_shapes : Dict[string, PartialShape]
                    Index of Output.

                Returns
                ----------
                reshape : void
             )");

    function.def(
        "reshape",
        [](ov::Model& self, const std::map<ov::Output<ov::Node>, ov::PartialShape>& partial_shapes) {
            self.reshape(partial_shapes);
        },
        py::arg("partial_shapes"),
        R"(
                Parameters
                ----------
                partial_shapes : Dict[Output, PartialShape]
                    Index of Output.

                Returns
                ----------
                reshape : void
             )");

    function.def("get_output_size",
                 &ov::Model::get_output_size,
                 R"(
                    Return the number of outputs for the function.

                    Returns
                    ----------
                    get_output_size : int
                        Number of outputs.
                 )");
    function.def("get_ops",
                 &ov::Model::get_ops,
                 R"(
                    Return ops used in the function.

                    Returns
                    ----------
                    get_ops : List[Node]
                        List of Nodes representing ops used in function.
                 )");
    function.def("get_ordered_ops",
                 &ov::Model::get_ordered_ops,
                 R"(
                    Return ops used in the function in topological order.

                    Returns
                    ----------
                    get_ordered_ops : List[Node]
                        List of sorted Nodes representing ops used in function.
                 )");
    function.def("get_output_op",
                 &ov::Model::get_output_op,
                 py::arg("index"),
                 R"(
                    Return the op that generates output i

                    Parameters
                    ----------
                    index : int
                        output index

                    Returns
                    ----------
                    get_output_op : Node
                        Node object that generates output i
                )");
    function.def("get_output_element_type",
                 &ov::Model::get_output_element_type,
                 py::arg("index"),
                 R"(
                    Return the element type of output i

                    Parameters
                    ----------
                    index : int
                        output index

                    Returns
                    ----------
                    get_output_op : Type
                        Type object of output i
                 )");
    function.def("get_output_shape",
                 &ov::Model::get_output_shape,
                 py::arg("index"),
                 R"(
                    Return the shape of element i

                    Parameters
                    ----------
                    index : int
                        element index

                    Returns
                    ----------
                    get_output_shape : Shape
                        Shape object of element i
                 )");
    function.def("get_output_partial_shape",
                 &ov::Model::get_output_partial_shape,
                 py::arg("index"),
                 R"(
                    Return the partial shape of element i

                    Parameters
                    ----------
                    index : int
                        element index

                    Returns
                    ----------
                    get_output_partial_shape : PartialShape
                        PartialShape object of element i
                 )");
    function.def("get_parameters",
                 &ov::Model::get_parameters,
                 R"(
                    Return the function parameters.

                    Returns
                    ----------
                    get_parameters : ParameterVector
                        ParameterVector containing function parameters.
                 )");
    function.def("get_results",
                 &ov::Model::get_results,
                 R"(
                    Return a list of function outputs.

                    Returns
                    ----------
                    get_results : ResultVector
                        ResultVector containing function parameters.
                 )");
    function.def("get_result",
                 &ov::Model::get_result,
                 R"(
                    Return single result.

                    Returns
                    ----------
                    get_result : Node
                        Node object representing result.
                 )");
    function.def("get_result_index",
                 (int64_t(ov::Model::*)(const ov::Output<ov::Node>&) const) & ov::Model::get_result_index,
                 py::arg("value"),
                 R"(
                    Return index of result.

                    Return -1 if `value` not matched.

                    Parameters
                    ----------
                    value : Output
                        Output containing Node

                    Returns
                    ----------
                    get_result_index : int
                        Index for value referencing it.
                 )");
    function.def("get_result_index",
                 (int64_t(ov::Model::*)(const ov::Output<const ov::Node>&) const) & ov::Model::get_result_index,
                 py::arg("value"),
                 R"(
                    Return index of result.

                    Return -1 if `value` not matched.

                    Parameters
                    ----------
                    value : Output
                        Output containing Node

                    Returns
                    ----------
                    get_result_index : int
                        Index for value referencing it.
                 )");

    function.def("get_name",
                 &ov::Model::get_name,
                 R"(
                    Get the unique name of the function.

                    Returns
                    ----------
                    get_name : str
                        String with a name of the function.
                 )");
    function.def("get_friendly_name",
                 &ov::Model::get_friendly_name,
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
                 &ov::Model::set_friendly_name,
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
                 &ov::Model::is_dynamic,
                 R"(
                    Returns true if any of the op's defined in the function
                    contains partial shape.

                    Returns
                    ----------
                    is_dynamic : bool
                 )");
    function.def("input", (ov::Output<ov::Node>(ov::Model::*)()) & ov::Model::input);

    function.def("input", (ov::Output<ov::Node>(ov::Model::*)(size_t)) & ov::Model::input, py::arg("index"));

    function.def("input",
                 (ov::Output<ov::Node>(ov::Model::*)(const std::string&)) & ov::Model::input,
                 py::arg("tensor_name"));

    function.def("input", (ov::Output<const ov::Node>(ov::Model::*)() const) & ov::Model::input);

    function.def("input",
                 (ov::Output<const ov::Node>(ov::Model::*)(size_t) const) & ov::Model::input,
                 py::arg("index"));

    function.def("input",
                 (ov::Output<const ov::Node>(ov::Model::*)(const std::string&) const) & ov::Model::input,
                 py::arg("tensor_name"));

    function.def("output", (ov::Output<ov::Node>(ov::Model::*)()) & ov::Model::output);

    function.def("output", (ov::Output<ov::Node>(ov::Model::*)(size_t)) & ov::Model::output, py::arg("index"));

    function.def("output",
                 (ov::Output<ov::Node>(ov::Model::*)(const std::string&)) & ov::Model::output,
                 py::arg("tensor_name"));

    function.def("output", (ov::Output<const ov::Node>(ov::Model::*)() const) & ov::Model::output);

    function.def("output",
                 (ov::Output<const ov::Node>(ov::Model::*)(size_t) const) & ov::Model::output,
                 py::arg("index"));

    function.def("output",
                 (ov::Output<const ov::Node>(ov::Model::*)(const std::string&) const) & ov::Model::output,
                 py::arg("tensor_name"));

    function.def(
        "add_outputs",
        [](ov::Model& self, py::handle& outputs) {
            int i = 0;
            std::vector<ov::Output<ov::Node>> new_outputs;
            py::list _outputs;
            if (!py::isinstance<py::list>(outputs)) {
                if (py::isinstance<py::str>(outputs)) {
                    _outputs.append(outputs.cast<py::str>());
                } else if (py::isinstance<py::tuple>(outputs)) {
                    _outputs.append(outputs.cast<py::tuple>());
                } else if (py::isinstance<ov::Output<ov::Node>>(outputs)) {
                    _outputs.append(outputs.cast<ov::Output<ov::Node>>());
                } else {
                    throw py::type_error("Incorrect type of a value to add as output.");
                }
            } else {
                _outputs = outputs.cast<py::list>();
            }

            for (py::handle output : _outputs) {
                ov::Output<ov::Node> out;
                if (py::isinstance<py::str>(_outputs[i])) {
                    out = self.add_output(output.cast<std::string>());
                } else if (py::isinstance<py::tuple>(output)) {
                    py::tuple output_tuple = output.cast<py::tuple>();
                    out = self.add_output(output_tuple[0].cast<std::string>(), output_tuple[1].cast<int>());
                } else if (py::isinstance<ov::Output<ov::Node>>(_outputs[i])) {
                    out = self.add_output(output.cast<ov::Output<ov::Node>>());
                } else {
                    throw py::type_error("Incorrect type of a value to add as output at index " + std::to_string(i) +
                                         ".");
                }
                new_outputs.emplace_back(out);
                i++;
            }
            return new_outputs;
        },
        py::arg("outputs"));

    function.def("replace_parameter",
                 &ov::Model::replace_parameter,
                 py::arg("parameter_index"),
                 py::arg("parameter"),
                 R"(
                    Replace the `parameter_index`th parameter of the function with `parameter`.

                    All users of the `parameter_index`th parameter are redirected to `parameter`, and the
                    `parameter_index`th entry in the function parameter list is replaced with `parameter`.

                    Parameters
                    ----------
                    parameter_index : int
                        The index of the parameter to replace.
                    parameter: op.Parameter
                        The parameter to substitute for the `parameter_index`th parameter.
        )");

    function.def(
        "get_parameter_index",
        (int64_t(ov::Model::*)(const std::shared_ptr<ov::op::v0::Parameter>&) const) & ov::Model::get_parameter_index,
        py::arg("parameter"),
        R"(
                    Return the index position of `parameter`.

                    Return -1 if parameter not matched.

                    Parameters
                    ----------
                    parameter : op.Parameter

                    Returns
                    ----------
                    get_parameter_index : int
                        Index for parameter
                 )");

    function.def(
        "evaluate",
        [](ov::Model& self,
           ov::runtime::TensorVector& output_tensors,
           const ov::runtime::TensorVector& input_tensors,
           PyRTMap evaluation_context) -> bool {
            return self.evaluate(output_tensors, input_tensors, evaluation_context);
        },
        py::arg("output_tensors"),
        py::arg("input_tensors"),
        py::arg("evaluation_context") = PyRTMap(),
        R"(
            Evaluate the function on inputs, putting results in outputs

            Parameters
            ----------
            output_tensors : List[op.Tensor]
                Tensors for the outputs to compute. One for each result
            input_tensors : List[op.Tensor]
                Tensors for the inputs. One for each inputs.
            evaluation_context: PyRTMap
                Storage of additional settings and attributes that can be used
                when evaluating the function. This additional information can be shared across nodes.

            Returns
            ----------
            evaluate : bool

        )");
    function.def("__repr__", [](const ov::Model& self) {
        std::string class_name = py::cast(self).get_type().attr("__name__").cast<std::string>();
        std::stringstream shapes_ss;
        for (size_t i = 0; i < self.get_output_size(); ++i) {
            if (i > 0) {
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

        auto* ngraph_function = static_cast<std::shared_ptr<ov::Model>*>(capsule_ptr);
        if (ngraph_function && *ngraph_function) {
            return *ngraph_function;
        } else {
            throw std::runtime_error("The provided capsule does not contain an ov::Model");
        }
    });
    function.def_static("to_capsule", [](std::shared_ptr<ov::Model>& ngraph_function) {
        // create a shared pointer on the heap before putting it in the capsule
        // this secures the lifetime of the object transferred by the capsule
        auto* sp_copy = new std::shared_ptr<ov::Model>(ngraph_function);

        // a destructor callback that will delete the heap allocated shared_ptr
        // when the capsule is destructed
        auto sp_deleter = [](PyObject* capsule) {
            auto* capsule_ptr = PyCapsule_GetPointer(capsule, CAPSULE_NAME);
            auto* function_sp = static_cast<std::shared_ptr<ov::Model>*>(capsule_ptr);
            if (function_sp) {
                delete function_sp;
            }
        };

        // put the shared_ptr in a new capsule under the same name as in "from_capsule"
        auto pybind_capsule = py::capsule(sp_copy, CAPSULE_NAME, sp_deleter);

        return pybind_capsule;
    });

    function.def_property_readonly("inputs", (std::vector<ov::Output<ov::Node>>(ov::Model::*)()) & ov::Model::inputs);
    function.def_property_readonly("outputs", (std::vector<ov::Output<ov::Node>>(ov::Model::*)()) & ov::Model::outputs);
    function.def_property_readonly("name", &ov::Model::get_name);
    function.def_property("friendly_name", &ov::Model::get_friendly_name, &ov::Model::set_friendly_name);
}
