// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/model.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/parameter.hpp"  // ov::op::v0::Parameter
#include "openvino/op/sink.hpp"
#include "openvino/op/util/variable.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/core/tensor.hpp"
#include "pyopenvino/graph/ops/result.hpp"
#include "pyopenvino/graph/ops/util/variable.hpp"
#include "pyopenvino/graph/rt_map.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

using PyRTMap = ov::RTMap;

PYBIND11_MAKE_OPAQUE(PyRTMap);

static void set_tensor_names(const ov::ParameterVector& parameters) {
    for (const auto& param : parameters) {
        ov::Output<ov::Node> p = param;
        if (p.get_node()->output(0).get_names().empty()) {
            std::unordered_set<std::string> p_names({p.get_node()->get_friendly_name()});
            p.get_node()->output(0).set_names(p_names);
        }
    }
}

static ov::SinkVector cast_to_sink_vector(const std::vector<std::shared_ptr<ov::Node>>& nodes) {
    ov::SinkVector sinks;
    for (const auto& node : nodes) {
        auto sink = std::dynamic_pointer_cast<ov::op::Sink>(node);
        OPENVINO_ASSERT(sink != nullptr, "Node {} is not instance of Sink");
        sinks.push_back(sink);
    }
    return sinks;
}

static std::vector<std::shared_ptr<ov::Node>> cast_to_node_vector(const ov::SinkVector& sinks) {
    std::vector<std::shared_ptr<ov::Node>> nodes;
    for (const auto& sink : sinks) {
        auto node = std::dynamic_pointer_cast<ov::Node>(sink);
        OPENVINO_ASSERT(node != nullptr, "Sink {} is not instance of Node");
        nodes.push_back(node);
    }
    return nodes;
}

// Assign operations created via Assign py binding have Variables which are not connected to
// ReadValue operations. This function attempts to resolve this situation by finding correct Variables
// for Assigns.
static void set_correct_variables_for_assign_ops(const std::shared_ptr<ov::Model>& model, const ov::SinkVector& sinks) {
    const auto& variables = model->get_variables();
    ov::op::util::VariableVector variables_to_delete;
    for (const auto& sink : sinks) {
        if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(sink)) {
            for (const auto& variable : variables) {
                auto info = variable->get_info();
                if (assign->get_variable_id() == info.variable_id && variable != assign->get_variable()) {
                    variables_to_delete.push_back(assign->get_variable());
                    assign->set_variable(variable);
                    break;
                }
            }
        }
    }

    for (const auto& var : variables_to_delete) {
        model->remove_variable(var);
    }
}

static ov::Output<ov::Node> output_from_handle(ov::Model& model, const py::handle& handle) {
    if (py::isinstance<py::int_>(handle)) {
        return model.input(handle.cast<size_t>());
    } else if (py::isinstance<py::str>(handle)) {
        return model.input(handle.cast<std::string>());
    } else if (py::isinstance<ov::Output<ov::Node>>(handle)) {
        return handle.cast<ov::Output<ov::Node>>();
    } else {
        throw py::type_error("Incorrect key type " + std::string(py::str(handle.get_type())) +
                             " to reshape a model, expected keys as openvino.runtime.Output, int or str.");
    }
}

static ov::PartialShape partial_shape_from_handle(const py::handle& handle) {
    if (py::isinstance<ov::PartialShape>(handle)) {
        return handle.cast<ov::PartialShape>();
    } else if (py::isinstance<py::list>(handle) || py::isinstance<py::tuple>(handle)) {
        return Common::partial_shape_from_list(handle.cast<py::list>());
    } else if (py::isinstance<py::str>(handle)) {
        return ov::PartialShape(handle.cast<std::string>());
    } else {
        throw py::type_error(
            "Incorrect value type " + std::string(py::str(handle.get_type())) +
            " to reshape a model, expected values as openvino.runtime.PartialShape, str, list or tuple.");
    }
}

static std::string string_from_handle(const py::handle& handle) {
    if (py::isinstance<py::str>(handle)) {
        return handle.cast<std::string>();
    } else {
        throw py::type_error("Incorrect key type " + std::string(py::str(handle.get_type())) +
                             " to reshape a model, expected values as str.");
    }
}

static std::unordered_map<std::string, ov::PartialShape> get_variables_shapes(const py::dict& variables_shapes) {
    std::unordered_map<std::string, ov::PartialShape> variables_shape_map;
    for (const auto& item : variables_shapes) {
        variables_shape_map.emplace(string_from_handle(item.first), partial_shape_from_handle(item.second));
    }
    return variables_shape_map;
}

void regclass_graph_Model(py::module m) {
    py::class_<ModelWrapper, std::shared_ptr<ModelWrapper>> model(m, "Model", py::module_local());
    model.doc() = "openvino.runtime.Model wraps ov::Model";

    model.def(py::init([](ModelWrapper& other) {
                  return other;
              }),
              py::arg("other"));

    model.def(py::init([](const ov::ResultVector& res,
                          const std::vector<std::shared_ptr<ov::Node>>& nodes,
                          const ov::ParameterVector& params,
                          const std::string& name) {
                  set_tensor_names(params);
                  const auto sinks = cast_to_sink_vector(nodes);
                  auto model = std::make_shared<ov::Model>(res, sinks, params, name);
                  set_correct_variables_for_assign_ops(model, sinks);
                  return ModelWrapper(model);
              }),
              py::arg("results"),
              py::arg("sinks"),
              py::arg("parameters"),
              py::arg("name") = "",
              R"(
                    Create user-defined Model which is a representation of a model.

                    :param results: List of results.
                    :type results: List[op.Result]
                    :param sinks: List of Nodes to be used as Sinks (e.g. Assign ops).
                    :type sinks: List[openvino.runtime.Node]
                    :param parameters: List of parameters.
                    :type parameters: List[op.Parameter]
                    :param name: String to set as model's friendly name.
                    :type name: str
                 )");

    // model.def(py::init([](const std::vector<std::shared_ptr<ov::Node>>& results,
    //                       const ov::ParameterVector& parameters,
    //                       const std::string& name) {
    //               set_tensor_names(parameters);
    //               return std::make_shared<ov::Model>(results, parameters, name);
    //           }),
    //           py::arg("results"),
    //           py::arg("parameters"),
    //           py::arg("name") = "",
    //           R"(
    //                 Create user-defined Model which is a representation of a model.

    //                 :param results: List of Nodes to be used as results.
    //                 :type results: List[openvino.runtime.Node]
    //                 :param parameters: List of parameters.
    //                 :type parameters:  List[op.Parameter]
    //                 :param name: String to set as model's friendly name.
    //                 :type name: str
    //              )");

    model.def(py::init([](const std::shared_ptr<ov::Node>& result,
                          const ov::ParameterVector& parameters,
                          const std::string& name) {
                  set_tensor_names(parameters);
                  return ModelWrapper(std::make_shared<ov::Model>(result, parameters, name));
              }),
              py::arg("result"),
              py::arg("parameters"),
              py::arg("name") = "",
              R"(
                    Create user-defined Model which is a representation of a model.

                    :param result: Node to be used as result.
                    :type result: openvino.runtime.Node
                    :param parameters: List of parameters.
                    :type parameters: List[op.Parameter]
                    :param name: String to set as model's friendly name.
                    :type name: str
                 )");

    model.def(
        py::init([](const ov::OutputVector& results, const ov::ParameterVector& parameters, const std::string& name)
        {
            set_tensor_names(parameters);
            return ModelWrapper(std::make_shared<ov::Model>(results, parameters, name));
        }),
        py::arg("results"),
        py::arg("parameters"),
        py::arg("name") = "",
        R"(
            Create user-defined Model which is a representation of a model

            :param results: List of outputs.
            :type results: List[openvino.runtime.Output]
            :param parameters: List of parameters.
            :type parameters: List[op.Parameter]
            :param name: String to set as model's friendly name.
            :type name: str
        )");

    model.def(py::init([](const ov::OutputVector& results,
                          const std::vector<std::shared_ptr<ov::Node>>& nodes,
                          const ov::ParameterVector& parameters,
                          const std::string& name) {
                  set_tensor_names(parameters);
                  const auto sinks = cast_to_sink_vector(nodes);
                  auto model = std::make_shared<ov::Model>(results, sinks, parameters, name);
                  set_correct_variables_for_assign_ops(model, sinks);
                  return ModelWrapper(model);
              }),
              py::arg("results"),
              py::arg("sinks"),
              py::arg("parameters"),
              py::arg("name") = "",
              R"(
            Create user-defined Model which is a representation of a model

            :param results: List of outputs.
            :type results: List[openvino.runtime.Output]
            :param sinks: List of Nodes to be used as Sinks (e.g. Assign ops).
            :type sinks: List[openvino.runtime.Node]
            :param name: String to set as model's friendly name.
            :type name: str
            )");
    model.def(py::init([](const ov::ResultVector& results,
                          const std::vector<std::shared_ptr<ov::Node>>& nodes,
                          const ov::ParameterVector& parameters,
                          const ov::op::util::VariableVector& variables,
                          const std::string& name) {
                  set_tensor_names(parameters);
                  const auto sinks = cast_to_sink_vector(nodes);
                  return ModelWrapper(std::make_shared<ov::Model>(results, sinks, parameters, variables, name));
              }),
              py::arg("results"),
              py::arg("sinks"),
              py::arg("parameters"),
              py::arg("variables"),
              py::arg("name") = "",
              R"(
            Create user-defined Model which is a representation of a model

            :param results: List of results.
            :type results: List[op.Result]
            :param sinks: List of Nodes to be used as Sinks (e.g. Assign ops).
            :type sinks: List[openvino.runtime.Node]
            :param parameters: List of parameters.
            :type parameters: List[op.Parameter]
            :param variables: List of variables.
            :type variables: List[op.util.Variable]
            :param name: String to set as model's friendly name.
            :type name: str
            )");

    model.def(py::init([](const ov::OutputVector& results,
                          const std::vector<std::shared_ptr<ov::Node>>& nodes,
                          const ov::ParameterVector& parameters,
                          const ov::op::util::VariableVector& variables,
                          const std::string& name) {
                  set_tensor_names(parameters);
                  const auto sinks = cast_to_sink_vector(nodes);
                  return ModelWrapper(std::make_shared<ov::Model>(results, sinks, parameters, variables, name));
              }),
              py::arg("results"),
              py::arg("sinks"),
              py::arg("parameters"),
              py::arg("variables"),
              py::arg("name") = "",
              R"(
            Create user-defined Model which is a representation of a model

            :param results: List of results.
            :type results: List[openvino.runtime.Output]
            :param sinks: List of Nodes to be used as Sinks (e.g. Assign ops).
            :type sinks: List[openvino.runtime.Node]
            :param variables: List of variables.
            :type variables: List[op.util.Variable]
            :param name: String to set as model's friendly name.
            :type name: str
        )");

    model.def(py::init([](const ov::ResultVector& results,
                          const ov::ParameterVector& parameters,
                          const ov::op::util::VariableVector& variables,
                          const std::string& name) {
                  set_tensor_names(parameters);
                  return ModelWrapper(std::make_shared<ov::Model>(results, parameters, variables, name));
              }),
              py::arg("results"),
              py::arg("parameters"),
              py::arg("variables"),
              py::arg("name") = "",
              R"(
            Create user-defined Model which is a representation of a model

            :param results: List of results.
            :type results: List[op.Result]
            :param parameters: List of parameters.
            :type parameters: List[op.Parameter]
            :param variables: List of variables.
            :type variables: List[op.util.Variable]
            :param name: String to set as model's friendly name.
            :type name: str
        )");

    model.def(py::init([](const ov::OutputVector& results,
                          const ov::ParameterVector& parameters,
                          const ov::op::util::VariableVector& variables,
                          const std::string& name) {
                  set_tensor_names(parameters);
                  return ModelWrapper(std::make_shared<ov::Model>(results, parameters, variables, name));
              }),
              py::arg("results"),
              py::arg("parameters"),
              py::arg("variables"),
              py::arg("name") = "",
              R"(
            Create user-defined Model which is a representation of a model

            :param results: List of results.
            :type results: List[openvino.runtime.Output]
            :param parameters: List of parameters.
            :type parameters: List[op.Parameter]
            :param name: String to set as model's friendly name.
            :type name: str
        )");

    model.def("validate_nodes_and_infer_types", [](ModelWrapper& self) {
        self.get_model().validate_nodes_and_infer_types();
    });

    model.def(
        "reshape",
        [](ModelWrapper& self, const ov::PartialShape& partial_shape, const py::dict& variables_shapes) {
            const auto new_variable_shapes = get_variables_shapes(variables_shapes);
            py::gil_scoped_release release;
            self.get_model().reshape(partial_shape, new_variable_shapes);
        },
        py::arg("partial_shape"),
        py::arg("variables_shapes") = py::dict(),
        R"(
                Reshape model input.

                The allowed types of keys in the `variables_shapes` dictionary is `str`.
                The allowed types of values in the `variables_shapes` are:

                (1) `openvino.runtime.PartialShape`
                (2) `list` consisting of dimensions
                (3) `tuple` consisting of dimensions
                (4) `str`, string representation of `openvino.runtime.PartialShape`

                When list or tuple are used to describe dimensions, each dimension can be written in form:

                (1) non-negative `int` which means static value for the dimension
                (2) `[min, max]`, dynamic dimension where `min` specifies lower bound and `max` specifies upper bound;
                the range includes both `min` and `max`; using `-1` for `min` or `max` means no known bound (3) `(min,
                max)`, the same as above (4) `-1` is a dynamic dimension without known bounds (4)
                `openvino.runtime.Dimension` (5) `str` using next syntax:
                    '?' - to define fully dynamic dimension
                    '1' - to define dimension which length is 1
                    '1..10' - to define bounded dimension
                    '..10' or '1..' to define dimension with only lower or only upper limit

                GIL is released while running this function.

                :param partial_shape: New shape.
                :type partial_shape: openvino.runtime.PartialShape
                :param variables_shapes: New shapes for variables
                :type variables_shapes: Dict[keys, values]
                :return : void
        )");

    model.def(
        "reshape",
        [](ModelWrapper& self, const py::list& partial_shape, const py::dict& variables_shapes) {
            const auto new_shape = Common::partial_shape_from_list(partial_shape);
            const auto new_variables_shapes = get_variables_shapes(variables_shapes);
            py::gil_scoped_release release;
            self.get_model().reshape(new_shape, new_variables_shapes);
        },
        py::arg("partial_shape"),
        py::arg("variables_shapes") = py::dict(),
        R"(
                Reshape model input.

                The allowed types of keys in the `variables_shapes` dictionary is `str`.
                The allowed types of values in the `variables_shapes` are:

                (1) `openvino.runtime.PartialShape`
                (2) `list` consisting of dimensions
                (3) `tuple` consisting of dimensions
                (4) `str`, string representation of `openvino.runtime.PartialShape`

                When list or tuple are used to describe dimensions, each dimension can be written in form:

                (1) non-negative `int` which means static value for the dimension
                (2) `[min, max]`, dynamic dimension where `min` specifies lower bound and `max` specifies upper bound;
                the range includes both `min` and `max`; using `-1` for `min` or `max` means no known bound (3) `(min,
                max)`, the same as above (4) `-1` is a dynamic dimension without known bounds (4)
                `openvino.runtime.Dimension` (5) `str` using next syntax:
                    '?' - to define fully dynamic dimension
                    '1' - to define dimension which length is 1
                    '1..10' - to define bounded dimension
                    '..10' or '1..' to define dimension with only lower or only upper limit

                GIL is released while running this function.

                :param partial_shape: New shape.
                :type partial_shape: list
                :param variables_shapes: New shapes for variables
                :type variables_shapes: Dict[keys, values]
                :return : void
        )");

    model.def(
        "reshape",
        [](ModelWrapper& self, const py::tuple& partial_shape, const py::dict& variables_shapes) {
            const auto new_shape = Common::partial_shape_from_list(partial_shape.cast<py::list>());
            const auto new_variables_shapes = get_variables_shapes(variables_shapes);
            py::gil_scoped_release release;
            self.get_model().reshape(new_shape, new_variables_shapes);
        },
        py::arg("partial_shape"),
        py::arg("variables_shapes") = py::dict(),
        R"(
                Reshape model input.

                The allowed types of keys in the `variables_shapes` dictionary is `str`.
                The allowed types of values in the `variables_shapes` are:

                (1) `openvino.runtime.PartialShape`
                (2) `list` consisting of dimensions
                (3) `tuple` consisting of dimensions
                (4) `str`, string representation of `openvino.runtime.PartialShape`

                When list or tuple are used to describe dimensions, each dimension can be written in form:

                (1) non-negative `int` which means static value for the dimension
                (2) `[min, max]`, dynamic dimension where `min` specifies lower bound and `max` specifies upper bound;
                the range includes both `min` and `max`; using `-1` for `min` or `max` means no known bound (3) `(min,
                max)`, the same as above (4) `-1` is a dynamic dimension without known bounds (4)
                `openvino.runtime.Dimension` (5) `str` using next syntax:
                    '?' - to define fully dynamic dimension
                    '1' - to define dimension which length is 1
                    '1..10' - to define bounded dimension
                    '..10' or '1..' to define dimension with only lower or only upper limit

                GIL is released while running this function.

                :param partial_shape: New shape.
                :type partial_shape: tuple
                :param variables_shapes: New shapes for variables
                :type variables_shapes: Dict[keys, values]
                :return : void
             )");

    model.def(
        "reshape",
        [](ModelWrapper& self, const std::string& partial_shape, const py::dict& variables_shapes) {
            const auto new_variables_shape = get_variables_shapes(variables_shapes);
            py::gil_scoped_release release;
            self.get_model().reshape(ov::PartialShape(partial_shape), new_variables_shape);
        },
        py::arg("partial_shape"),
        py::arg("variables_shapes") = py::dict(),
        R"(
                Reshape model input.

                The allowed types of keys in the `variables_shapes` dictionary is `str`.
                The allowed types of values in the `variables_shapes` are:

                (1) `openvino.runtime.PartialShape`
                (2) `list` consisting of dimensions
                (3) `tuple` consisting of dimensions
                (4) `str`, string representation of `openvino.runtime.PartialShape`

                When list or tuple are used to describe dimensions, each dimension can be written in form:

                (1) non-negative `int` which means static value for the dimension
                (2) `[min, max]`, dynamic dimension where `min` specifies lower bound and `max` specifies upper
                bound; the range includes both `min` and `max`; using `-1` for `min` or `max` means no known bound
                (3) `(min, max)`, the same as above (4) `-1` is a dynamic dimension without known bounds (4)
                `openvino.runtime.Dimension` (5) `str` using next syntax:
                    '?' - to define fully dynamic dimension
                    '1' - to define dimension which length is 1
                    '1..10' - to define bounded dimension
                    '..10' or '1..' to define dimension with only lower or only upper limit

                GIL is released while running this function.

                :param partial_shape: New shape.
                :type partial_shape: str
                :param variables_shapes: New shapes for variables
                :type variables_shapes: Dict[keys, values]
                :return : void
        )");

    model.def(
        "reshape",
        [](ModelWrapper& self, const py::dict& partial_shapes, const py::dict& variables_shapes) {
            std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes;
            auto& model = self.get_model();
            for (const auto& item : partial_shapes) {
                new_shapes.emplace_hint(new_shapes.end(),
                                        output_from_handle(model, item.first),
                                        partial_shape_from_handle(item.second));
            }
            const auto new_variables_shapes = get_variables_shapes(variables_shapes);
            py::gil_scoped_release release;
            model.reshape(new_shapes, new_variables_shapes);
        },
        py::arg("partial_shapes"),
        py::arg("variables_shapes") = py::dict(),
        R"( Reshape model inputs.

            The allowed types of keys in the `partial_shapes` dictionary are:

            (1) `int`, input index
            (2) `str`, input tensor name
            (3) `openvino.runtime.Output`

            The allowed types of values in the `partial_shapes` are:

            (1) `openvino.runtime.PartialShape`
            (2) `list` consisting of dimensions
            (3) `tuple` consisting of dimensions
            (4) `str`, string representation of `openvino.runtime.PartialShape`

            When list or tuple are used to describe dimensions, each dimension can be written in form:

            (1) non-negative `int` which means static value for the dimension
            (2) `[min, max]`, dynamic dimension where `min` specifies lower bound and `max` specifies upper bound; the range includes both `min` and `max`; using `-1` for `min` or `max` means no known bound
            (3) `(min, max)`, the same as above
            (4) `-1` is a dynamic dimension without known bounds
            (4) `openvino.runtime.Dimension`
            (5) `str` using next syntax:
                '?' - to define fully dynamic dimension
                '1' - to define dimension which length is 1
                '1..10' - to define bounded dimension
                '..10' or '1..' to define dimension with only lower or only upper limit

            The allowed types of keys in the `variables_shapes` dictionary is `str`.
            The allowed types of values in the `variables_shapes` are:

            (1) `openvino.runtime.PartialShape`
            (2) `list` consisting of dimensions
            (3) `tuple` consisting of dimensions
            (4) `str`, string representation of `openvino.runtime.PartialShape`

            When list or tuple are used to describe dimensions, each dimension can be written in form:

            (1) non-negative `int` which means static value for the dimension
            (2) `[min, max]`, dynamic dimension where `min` specifies lower bound and `max` specifies upper bound;
            the range includes both `min` and `max`; using `-1` for `min` or `max` means no known bound (3) `(min,
            max)`, the same as above (4) `-1` is a dynamic dimension without known bounds (4)
            `openvino.runtime.Dimension` (5) `str` using next syntax:
                '?' - to define fully dynamic dimension
                '1' - to define dimension which length is 1
                '1..10' - to define bounded dimension
                '..10' or '1..' to define dimension with only lower or only upper limit

            Reshape model inputs.

            GIL is released while running this function.

            :param partial_shapes: New shapes.
            :type partial_shapes: Dict[keys, values]
            :param variables_shapes: New shapes for variables
            :type variables_shapes: Dict[keys, values]
        )");

    model.def(
        "get_output_size",
        [](ModelWrapper& self) {
            return self.get_model().get_output_size();
        },
        R"(
            Return the number of outputs for the model.

            :return: Number of outputs.
            :rtype: int
        )");
    model.def(
        "get_ops",
        [](ModelWrapper& self) {
            return self.get_model().get_ops();
        },
        R"(
            Return ops used in the model.

            :return: List of Nodes representing ops used in model.
            :rtype: List[openvino.runtime.Node]
        )");
    model.def(
        "get_ordered_ops",
        [](ModelWrapper& self) {
            return self.get_model().get_ordered_ops();
        },
        R"(
            Return ops used in the model in topological order.

            :return: List of sorted Nodes representing ops used in model.
            :rtype: List[openvino.runtime.Node]
        )");
    model.def(
        "get_output_op",
        [](ModelWrapper& self, size_t i) {
            return self.get_model().get_output_op(i);
        },
        py::arg("index"),
        R"(
            Return the op that generates output i

            :param index: output index
            :type index: output index
            :return: Node object that generates output i
            :rtype: openvino.runtime.Node
        )");
    model.def(
        "get_output_element_type",
        [](ModelWrapper& self, size_t i) {
            return self.get_model().get_output_element_type(i);
        },
        py::arg("index"),
        R"(
            Return the element type of output i

            :param index: output index
            :type index: int
            :return: Type object of output i
            :rtype: openvino.runtime.Type
        )");
    model.def(
        "get_output_shape",
        [](ModelWrapper& self, size_t i) {
            return self.get_model().get_output_shape(i);
        },
        py::arg("index"),
        R"(
            Return the shape of element i

            :param index: element index
            :type index: int
            :return: Shape object of element i
            :rtype: openvino.runtime.Shape
        )");
    model.def(
        "get_output_partial_shape",
        [](ModelWrapper& self, size_t i) {
            return self.get_model().get_output_partial_shape(i);
        },
        py::arg("index"),
        R"(
            Return the partial shape of element i

            :param index: element index
            :type index: int
            :return: PartialShape object of element i
            :rtype: openvino.runtime.PartialShape
        )");
    model.def(
        "get_parameters",
        [](ModelWrapper& self) {
            return self.get_model().get_parameters();
        },
        R"(
            Return the model parameters.
            
            :return: a list of model's parameters.
            :rtype: List[op.Parameter]
        )");
    model.def_property_readonly(
        "parameters",
        [](ModelWrapper& self) {
            return self.get_model().get_parameters();
        },
        R"(
            Return the model parameters.
            
            :return: a list of model's parameters.
            :rtype: List[op.Parameter]
        )");
    model.def(
        "get_results",
        [](ModelWrapper& self) {
            return self.get_model().get_results();
        },
        R"(
            Return a list of model outputs.

            :return: a list of model's result nodes.
            :rtype: List[op.Result]
        )");
    model.def_property_readonly(
        "results",
        [](ModelWrapper& self) {
            return self.get_model().get_results();
        },
        R"(
            Return a list of model outputs.

            :return: a list of model's result nodes.
            :rtype: List[op.Result]
        )");
    model.def(
        "get_result",
        [](ModelWrapper& self) {
            return self.get_model().get_result();
        },
        R"(
            Return single result.

            :return: Node object representing result.
            :rtype: op.Result
        )");
    model.def_property_readonly(
        "result",
        [](ModelWrapper& self) {
            return self.get_model().get_result();
        },
        R"(
            Return single result.

            :return: Node object representing result.
            :rtype: op.Result
        )");
    model.def(
        "get_result_index",
        [](ModelWrapper& self, const ov::Output<ov::Node>& value) {
            return self.get_model().get_result_index(value);
        },
        py::arg("value"),
        R"(
            Return index of result.

            Return -1 if `value` not matched.

            :param value: Output containing Node
            :type value: openvino.runtime.Output
            :return: Index for value referencing it.
            :rtype: int
        )");
    model.def(
        "get_result_index",
        [](ModelWrapper& self, const ov::Output<const ov::Node>& value) {
            return self.get_model().get_result_index(value);
        },
        py::arg("value"),
        R"(
            Return index of result.

            Return -1 if `value` not matched.

            :param value: Output containing Node
            :type value: openvino.runtime.Output
            :return: Index for value referencing it.
            :rtype: int
        )");

    model.def(
        "get_name",
        [](ModelWrapper& self) {
            return self.get_model().get_name();
        },
        R"(
            Get the unique name of the model.

            :return: String with a name of the model.
            :rtype: str
        )");
    model.def(
        "get_friendly_name",
        [](ModelWrapper& self) {
            return self.get_model().get_friendly_name();
        },
        R"(
            Gets the friendly name for a model. If no
            friendly name has been set via set_friendly_name
            then the model's unique name is returned.

            :return: String with a friendly name of the model.
            :rtype: str
        )");
    model.def(
        "set_friendly_name",
        [](ModelWrapper& self, const std::string& name) {
            self.get_model().set_friendly_name(name);
        },
        py::arg("name"),
        R"(
            Sets a friendly name for a model. This does
            not overwrite the unique name of the model and
            is retrieved via get_friendly_name(). Used mainly
            for debugging.

            :param name: String to set as the friendly name.
            :type name: str
        )");
    model.def(
        "is_dynamic",
        [](ModelWrapper& self) {
            return self.get_model().is_dynamic();
        },
        R"(
            Returns true if any of the op's defined in the model
            contains partial shape.

            :rtype: bool
        )");
    model.def_property_readonly(
        "dynamic",
        [](ModelWrapper& self) {
            return self.get_model().is_dynamic();
        },
        R"(
            Returns true if any of the op's defined in the model
            contains partial shape.

            :rtype: bool
        )");
    model.def("input", [](ModelWrapper& self) {
        return self.get_model().input();
    });

    model.def(
        "input",
        [](ModelWrapper& self, size_t i) {
            return self.get_model().input(i);
        },
        py::arg("index"));

    model.def(
        "input",
        [](ModelWrapper& self, const std::string& tensor_name) {
            return self.get_model().input(tensor_name);
        },
        py::arg("tensor_name"));

    model.def("input", [](const ModelWrapper& self) {
        return self.get_model().input();
    });

    model.def(
        "input",
        [](const ModelWrapper& self, size_t i) {
            return self.get_model().input(i);
        },
        py::arg("index"));

    model.def(
        "input",
        [](const ModelWrapper& self, const std::string& tensor_name) {
            return self.get_model().input(tensor_name);
        },
        py::arg("tensor_name"));

    model.def("output", [](ModelWrapper& self) {
        return self.get_model().output();
    });

    model.def(
        "output",
        [](ModelWrapper& self, size_t i) {
            return self.get_model().output(i);
        },
        py::arg("index"));

    model.def(
        "output",
        [](ModelWrapper& self, const std::string& tensor_name) {
            return self.get_model().output(tensor_name);
        },
        py::arg("tensor_name"));

    model.def("output", [](const ModelWrapper& self) {
        return self.get_model().output();
    });

    model.def(
        "output",
        [](const ModelWrapper& self, size_t i) {
            return self.get_model().output(i);
        },
        py::arg("index"));

    model.def(
        "output",
        [](const ModelWrapper& self, const std::string& tensor_name) {
            return self.get_model().output(tensor_name);
        },
        py::arg("tensor_name"));

    model.def(
        "add_outputs",
        [](ModelWrapper& self, py::handle& outputs) {
            auto& model = self.get_model();
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
                    out = model.add_output(output.cast<std::string>());
                } else if (py::isinstance<py::tuple>(output)) {
                    py::tuple output_tuple = output.cast<py::tuple>();
                    out = model.add_output(output_tuple[0].cast<std::string>(), output_tuple[1].cast<int>());
                } else if (py::isinstance<ov::Output<ov::Node>>(_outputs[i])) {
                    out = model.add_output(output.cast<ov::Output<ov::Node>>());
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

    model.def(
        "replace_parameter",
        [](ModelWrapper& self, size_t parameter_index, const std::shared_ptr<ov::op::v0::Parameter>& parameter) {
            self.get_model().replace_parameter(parameter_index, parameter);
        },
        py::arg("parameter_index"),
        py::arg("parameter"),
        R"(
            Replace the `parameter_index` parameter of the model with `parameter`

            All users of the `parameter_index` parameter are redirected to `parameter` , and the
            `parameter_index` entry in the model parameter list is replaced with `parameter`

            :param parameter_index: The index of the parameter to replace.
            :type parameter_index: int
            :param parameter: The parameter to substitute for the `parameter_index` parameter.
            :type parameter: op.Parameter
        )");

    model.def(
        "get_parameter_index",
        [](ModelWrapper& self, const std::shared_ptr<ov::op::v0::Parameter>& parameter) {
            return self.get_model().get_parameter_index(parameter);
        },
        py::arg("parameter"),
        R"(
            Return the index position of `parameter`

            Return -1 if parameter not matched.

            :param parameter: Parameter, which index is to be found.
            :type parameter: op.Parameter
            :return: Index for parameter
            :rtype: int
        )");

    model.def(
        "remove_result",
        [](ModelWrapper& self, const std::shared_ptr<ov::op::v0::Result>& result) {
            self.get_model().remove_result(result);
        },
        py::arg("result"),
        R"(
            Delete Result node from the list of results. Method will not delete node from graph.

            :param result: Result node to delete.
            :type result: op.Result
        )");

    model.def(
        "remove_parameter",
        [](ModelWrapper& self, const std::shared_ptr<ov::op::v0::Parameter>& parameter) {
            self.get_model().remove_parameter(parameter);
        },
        py::arg("parameter"),
        R"(
            Delete Parameter node from the list of parameters. Method will not delete node from graph. 
            You need to replace Parameter with other operation manually.

            Attention: Indexing of parameters can be changed.

            Possible use of method is to replace input by variable. For it the following steps should be done:
            * `Parameter` node should be replaced by `ReadValue`
            * call remove_parameter(param) to remove input from the list
            * check if any parameter indexes are saved/used somewhere, update it for all inputs because indexes can be changed
            * call graph validation to check all changes

            :param parameter: Parameter node to delete.
            :type parameter: op.Parameter
        )");

    model.def(
        "remove_sink",
        [](ModelWrapper& self, const py::object& node) {
            auto& model = self.get_model();
            if (py::isinstance<ov::op::v6::Assign>(node)) {
                auto sink = std::dynamic_pointer_cast<ov::op::Sink>(node.cast<std::shared_ptr<ov::op::v6::Assign>>());
                model.remove_sink(sink);
            } else if (py::isinstance<ov::Node>(node)) {
                auto sink = std::dynamic_pointer_cast<ov::op::Sink>(node.cast<std::shared_ptr<ov::Node>>());
                model.remove_sink(sink);
            } else {
                throw py::type_error("Incorrect argument type. Sink node is expected as an argument.");
            }
        },
        py::arg("sink"),
        R"(
            Delete sink node from the list of sinks. Method doesn't delete node from graph.

            :param sink: Sink to delete.
            :type sink: openvino.runtime.Node
        )");

    model.def(
        "remove_variable",
        [](ModelWrapper& self, const ov::op::util::Variable::Ptr& variable) {
            self.get_model().remove_variable(variable);
        },
        py::arg("variable"),
        R"(
            Delete variable from the list of variables.
            Method doesn't delete nodes that used this variable from the graph.

            :param variable:  Variable to delete.
            :type variable: op.util.Variable
        )");

    model.def(
        "add_parameters",
        [](ModelWrapper& self, const ov::ParameterVector& params) {
            self.get_model().add_parameters(params);
        },
        py::arg("parameters"),
        R"(
            Add new Parameter nodes to the list.

            Method doesn't change or validate graph, it should be done manually.
            For example, if you want to replace `ReadValue` node by `Parameter`, you should do the
            following steps:
            * replace node `ReadValue` by `Parameter` in graph
            * call add_parameter() to add new input to the list
            * call graph validation to check correctness of changes

            :param parameter: new Parameter nodes.
            :type parameter: List[op.Parameter]
        )");

    model.def(
        "add_results",
        [](ModelWrapper& self, const ov::ResultVector& results) {
            self.get_model().add_results(results);
        },
        py::arg("results"),
        R"(
            Add new Result nodes to the list.
            
            Method doesn't validate graph, it should be done manually after all changes.

            :param results: new Result nodes.
            :type results: List[op.Result]
        )");

    model.def(
        "add_sinks",
        [](ModelWrapper& self, py::list& sinks) {
            ov::SinkVector sinks_cpp;
            for (py::handle sink : sinks) {
                auto sink_cpp =
                    std::dynamic_pointer_cast<ov::op::Sink>(sink.cast<std::shared_ptr<ov::op::v6::Assign>>());
                OPENVINO_ASSERT(sink_cpp != nullptr, "Assign {} is not instance of Sink");
                sinks_cpp.push_back(sink_cpp);
            }
            self.get_model().add_sinks(sinks_cpp);
        },
        py::arg("sinks"),
        R"(
            Add new sink nodes to the list.
            
            Method doesn't validate graph, it should be done manually after all changes.

            :param sinks: new sink nodes.
            :type sinks: List[openvino.runtime.Node]
        )");

    model.def(
        "add_variables",
        [](ModelWrapper& self, const ov::op::util::VariableVector& variables) {
            self.get_model().add_variables(variables);
        },
        py::arg("variables"),
        R"(
            Add new variables to the list. 
            
            Method doesn't validate graph, it should be done manually after all changes.

            :param variables: new variables to add.
            :type variables: List[op.util.Variable]
        )");

    model.def(
        "get_variables",
        [](ModelWrapper& self) {
            return self.get_model().get_variables();
        },
        R"(
            Return a list of model's variables.
            
            :return: a list of model's variables.
            :rtype: List[op.util.Variable]
        )");

    model.def_property_readonly(
        "variables",
        [](ModelWrapper& self) {
            return self.get_model().get_variables();
        },
        R"(
            Return a list of model's variables.
            
            :return: a list of model's variables.
            :rtype: List[op.util.Variable]
        )");

    model.def(
        "get_variable_by_id",
        [](ModelWrapper& self, const std::string& variable_id) {
            return self.get_model().get_variable_by_id(variable_id);
        },
        R"(
            Return a variable by specified variable_id.
            
            :param variable_id: a variable id to get variable node.
            :type variable_id: str
            :return: a variable node.
            :rtype: op.util.Variable
        )");

    model.def(
        "get_sinks",
        [](ModelWrapper& self) {
            auto sinks = self.get_model().get_sinks();
            return cast_to_node_vector(sinks);
        },
        R"(
            Return a list of model's sinks.

            :return: a list of model's sinks.
            :rtype: List[openvino.runtime.Node]
        )");

    model.def_property_readonly(
        "sinks",
        [](ModelWrapper& self) {
            auto sinks = self.get_model().get_sinks();
            return cast_to_node_vector(sinks);
        },
        R"(
            Return a list of model's sinks.

            :return: a list of model's sinks.
            :rtype: List[openvino.runtime.Node]
        )");

    model.def(
        "evaluate",
        [](ModelWrapper& self,
           ov::TensorVector& output_tensors,
           const ov::TensorVector& input_tensors,
           PyRTMap evaluation_context) -> bool {
            return self.get_model().evaluate(output_tensors, input_tensors, evaluation_context);
        },
        py::arg("output_tensors"),
        py::arg("input_tensors"),
        py::arg("evaluation_context") = PyRTMap(),
        R"(
            Evaluate the model on inputs, putting results in outputs

            :param output_tensors: Tensors for the outputs to compute. One for each result
            :type output_tensors: List[openvino.runtime.Tensor]
            :param input_tensors: Tensors for the inputs. One for each inputs.
            :type input_tensors: List[openvino.runtime.Tensor]
            :param evaluation_context: Storage of additional settings and attributes that can be used
                                       when evaluating the model. This additional information can be
                                       shared across nodes.
            :type evaluation_context: openvino.runtime.RTMap
            :rtype: bool
        )");

    model.def(
        "clone",
        [](const ModelWrapper& self) {
            return self.get_model().clone();
        },
        R"(
            Return a copy of self.
            :return: A copy of self.
            :rtype: openvino.runtime.Model
        )");

    model.def("__repr__", [](const ModelWrapper& self) {
        const auto& model = self.get_model();
        std::string class_name = Common::get_class_name(model);

        auto inputs_str = Common::docs::container_to_string(model.inputs(), ",\n");
        auto outputs_str = Common::docs::container_to_string(model.outputs(), ",\n");

        return "<" + class_name + ": '" + model.get_friendly_name() + "'\ninputs[\n" + inputs_str + "\n]\noutputs[\n" +
               outputs_str + "\n]>";
    });

    model.def("__copy__", [](ModelWrapper& self) {
        throw py::type_error("Cannot copy 'openvino.runtime.Model. Please, use deepcopy instead.");
    });

    model.def("__del__", [](ModelWrapper& self) {
        self.erase();
        std::cout << "Hey, im deleted too! Refcount is " << std::endl;
    });

    model.def(
        "get_rt_info",
        [](ModelWrapper& self) {
            return self.get_model().get_rt_info();
        },
        py::return_value_policy::reference_internal,
        R"(
            Returns PyRTMap which is a dictionary of user defined runtime info.

            :return: A dictionary of user defined data.
            :rtype: openvino.runtime.RTMap
        )");
    model.def(
        "get_rt_info",
        [](const ModelWrapper& self, const py::list& path) -> py::object {
            std::vector<std::string> cpp_args(path.size());
            for (size_t i = 0; i < path.size(); i++) {
                cpp_args[i] = path[i].cast<std::string>();
            }
            return py::cast(self.get_model().get_rt_info<ov::Any>(cpp_args));
        },
        py::arg("path"),
        R"(
            Returns runtime attribute as a OVAny object.

            :param path: List of strings which defines a path to runtime info.
            :type path: List[str]

            :return: A runtime attribute.
            :rtype: openvino.runtime.OVAny
            )");
    model.def(
        "get_rt_info",
        [](const ModelWrapper& self, const py::str& path) -> py::object {
            return py::cast(self.get_model().get_rt_info<ov::Any>(path.cast<std::string>()));
        },
        py::arg("path"),
        R"(
            Returns runtime attribute as a OVAny object.

            :param path: List of strings which defines a path to runtime info.
            :type path: str

            :return: A runtime attribute.
            :rtype: openvino.runtime.OVAny
        )");
    model.def(
        "has_rt_info",
        [](const ModelWrapper& self, const py::list& path) -> bool {
            std::vector<std::string> cpp_args(path.size());
            for (size_t i = 0; i < path.size(); i++) {
                cpp_args[i] = path[i].cast<std::string>();
            }
            return self.get_model().has_rt_info(cpp_args);
        },
        py::arg("path"),
        R"(
            Checks if given path exists in runtime info of the model.

            :param path: List of strings which defines a path to runtime info.
            :type path: List[str]

            :return: `True` if path exists, otherwise `False`.
            :rtype: bool
        )");
    model.def(
        "has_rt_info",
        [](const ModelWrapper& self, const py::str& path) -> bool {
            return self.get_model().has_rt_info(path.cast<std::string>());
        },
        py::arg("path"),
        R"(
            Checks if given path exists in runtime info of the model.

            :param path: List of strings which defines a path to runtime info.
            :type path: str

            :return: `True` if path exists, otherwise `False`.
            :rtype: bool
        )");
    model.def(
        "set_rt_info",
        [](ModelWrapper& self, const py::object& obj, const py::list& path) -> void {
            std::vector<std::string> cpp_args(path.size());
            for (size_t i = 0; i < path.size(); i++) {
                cpp_args[i] = path[i].cast<std::string>();
            }
            self.get_model().set_rt_info<ov::Any>(Common::utils::py_object_to_any(obj), cpp_args);
        },
        py::arg("obj"),
        py::arg("path"),
        R"(
            Add value inside runtime info

            :param obj: value for the runtime info
            :type obj: py:object
            :param path: List of strings which defines a path to runtime info.
            :type path: List[str]
        )");
    model.def(
        "set_rt_info",
        [](ModelWrapper& self, const py::object& obj, const py::str& path) -> void {
            self.get_model().set_rt_info<ov::Any>(Common::utils::py_object_to_any(obj), path.cast<std::string>());
        },
        py::arg("obj"),
        py::arg("path"),
        R"(
            Add value inside runtime info

            :param obj: value for the runtime info
            :type obj: Any
            :param path: String which defines a path to runtime info.
            :type path: str
        )");

    model.def(
        "_get_raw_address",
        [](ModelWrapper& self) {
            return reinterpret_cast<uint64_t>(&self.get_model());
        },
        R"(
            Returns a raw address of the Model object from C++.
            
            Use this function in order to compare underlying C++ addresses instead of using `__eq__` in Python.

            :return: a raw address of the Model object.
            :rtype: int
        )");

    model.def_property_readonly("inputs", [](ModelWrapper& self) {
        return self.get_model().inputs();
    });
    model.def_property_readonly("outputs", [](ModelWrapper& self) {
        return self.get_model().outputs();
    });
    model.def_property_readonly("name", [](ModelWrapper& self) {
        return self.get_model().get_name();
    });
    model.def_property_readonly(
        "rt_info",
        [](ModelWrapper& self) {
            return self.get_model().get_rt_info();
        },
        py::return_value_policy::reference_internal);
    model.def_property(
        "friendly_name",
        [](ModelWrapper& self) {
            return self.get_model().get_friendly_name();
        },
        [](ModelWrapper& self, const std::string& name) {
            self.get_model().set_friendly_name(name);
        });
}
