// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/manager.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_frontend_InputModel(py::module m) {
    py::class_<ov::frontend::InputModel, std::shared_ptr<ov::frontend::InputModel>> im(m,
                                                                                       "InputModel",
                                                                                       py::dynamic_attr());
    im.doc() = "openvino.frontend.InputModel wraps ov::frontend::InputModel";

    im.def("get_place_by_tensor_name",
           &ov::frontend::InputModel::get_place_by_tensor_name,
           py::arg("tensor_name"),
           R"(
                Returns a tensor place by a tensor name following framework conventions, or
                nullptr if a tensor with this name doesn't exist.

                Parameters
                ----------
                tensor_name : str
                    Name of tensor.

                Returns
                ----------
                get_place_by_tensor_name : Place
                    Tensor place corresponding to specified tensor name.
             )");

    im.def("get_place_by_operation_name",
           &ov::frontend::InputModel::get_place_by_operation_name,
           py::arg("operation_name"),
           R"(
                Returns an operation place by an operation name following framework conventions, or
                nullptr if an operation with this name doesn't exist.

                Parameters
                ----------
                operation_name : str
                    Name of operation.

                Returns
                ----------
                get_place_by_operation_name : Place
                    Place representing operation.
             )");

    im.def("get_place_by_operation_name_and_input_port",
           &ov::frontend::InputModel::get_place_by_operation_name_and_input_port,
           py::arg("operation_name"),
           py::arg("input_port_index"),
           R"(
                Returns an input port place by operation name and appropriate port index.

                Parameters
                ----------
                operation_name : str
                    Name of operation.

                input_port_index : int
                    Index of input port for this operation.

                Returns
                ----------
                get_place_by_operation_name_and_input_port : Place
                    Place representing input port of operation.
             )");

    im.def("get_place_by_operation_name_and_output_port",
           &ov::frontend::InputModel::get_place_by_operation_name_and_output_port,
           py::arg("operation_name"),
           py::arg("output_port_index"),
           R"(
                Returns an output port place by operation name and appropriate port index.

                Parameters
                ----------
                operation_name : str
                    Name of operation.

                output_port_index : int
                    Index of output port for this operation.

                Returns
                ----------
                get_place_by_operation_name_and_output_port : Place
                    Place representing output port of operation.
             )");

    im.def("set_name_for_tensor",
           &ov::frontend::InputModel::set_name_for_tensor,
           py::arg("tensor"),
           py::arg("new_name"),
           R"(
                Sets name for tensor. Overwrites existing names of this place.

                Parameters
                ----------
                tensor : Place
                    Tensor place.

                new_name : str
                    New name for this tensor.
            )");

    im.def("add_name_for_tensor",
           &ov::frontend::InputModel::add_name_for_tensor,
           py::arg("tensor"),
           py::arg("new_name"),
           R"(
                Adds new name for tensor

                Parameters
                ----------
                tensor : Place
                    Tensor place.

                new_name : str
                    New name to be added to this place.
            )");

    im.def("set_name_for_operation",
           &ov::frontend::InputModel::set_name_for_operation,
           py::arg("operation"),
           py::arg("new_name"),
           R"(
                Adds new name for tensor.

                Parameters
                ----------
                operation : Place
                    Operation place.

                new_name : str
                    New name for this operation.
            )");

    im.def("free_name_for_tensor",
           &ov::frontend::InputModel::free_name_for_tensor,
           py::arg("name"),
           R"(
                Unassign specified name from tensor place(s).

                Parameters
                ----------
                name : str
                    Name of tensor.
            )");

    im.def("free_name_for_operation",
           &ov::frontend::InputModel::free_name_for_operation,
           py::arg("name"),
           R"(
                Unassign specified name from operation place(s).

                Parameters
                ----------
                name : str
                    Name of operation.
            )");

    im.def("set_name_for_dimension",
           &ov::frontend::InputModel::set_name_for_dimension,
           py::arg("place"),
           py::arg("dim_index"),
           py::arg("dim_name"),
           R"(
                Set name for a particular dimension of a place (e.g. batch dimension).

                Parameters
                ----------
                place : Place
                    Model's place.

                dim_index : int
                    Dimension index.

                dim_name : str
                    Name to assign on this dimension.
            )");

    im.def("cut_and_add_new_input",
           &ov::frontend::InputModel::cut_and_add_new_input,
           py::arg("place"),
           py::arg("new_name") = std::string(),
           R"(
                Cut immediately before this place and assign this place as new input; prune
                all nodes that don't contribute to any output.

                Parameters
                ----------
                place : Place
                    New place to be assigned as input.

                new_name_optional : str
                    Optional new name assigned to this input place.
            )");

    im.def("cut_and_add_new_output",
           &ov::frontend::InputModel::cut_and_add_new_output,
           py::arg("place"),
           py::arg("new_name") = std::string(),
           R"(
                Cut immediately before this place and assign this place as new output; prune
                all nodes that don't contribute to any output.

                Parameters
                ----------
                place : Place
                    New place to be assigned as output.

                new_name_optional : str
                    Optional new name assigned to this output place.
            )");

    im.def("add_output",
           &ov::frontend::InputModel::add_output,
           py::arg("place"),
           R"(
                Assign this place as new output or add necessary nodes to represent a new output.

                Parameters
                ----------
                place : Place
                    Anchor point to add an output.
            )");

    im.def("remove_output",
           &ov::frontend::InputModel::remove_output,
           py::arg("place"),
           R"(
                Removes any sinks directly attached to this place with all inbound data flow
                if it is not required by any other output.

                Parameters
                ----------
                place : Place
                    Model place
            )");

    im.def("set_partial_shape",
           &ov::frontend::InputModel::set_partial_shape,
           py::arg("place"),
           py::arg("shape"),
           R"(
                Defines all possible shape that may be used for this place; place should be
                uniquely refer to some data. This partial shape will be converted to corresponding
                shape of results ngraph nodes and will define shape inference when the model is
                converted to ngraph.

                Parameters
                ----------
                place : Place
                    Model place.

                shape : PartialShape
                    Partial shape for this place.
            )");

    im.def("get_partial_shape",
           &ov::frontend::InputModel::get_partial_shape,
           py::arg("place"),
           R"(
                Returns current partial shape used for this place.

                Parameters
                ----------
                place : Place
                    Model place

                Returns
                ----------
                get_partial_shape : PartialShape
                    Partial shape for this place.
            )");

    im.def("get_inputs",
           &ov::frontend::InputModel::get_inputs,
           R"(
                Returns all inputs for a model.

                Returns
                ----------
                get_inputs : List[Place]
                    A list of input places.
            )");

    im.def("get_outputs",
           &ov::frontend::InputModel::get_outputs,
           R"(
                Returns all outputs for a model. An output is a terminal place in a graph where data escapes the flow.

                Returns
                ----------
                get_outputs : List[Place]
                    A list of output places
            )");

    im.def("extract_subgraph",
           &ov::frontend::InputModel::extract_subgraph,
           py::arg("inputs"),
           py::arg("outputs"),
           R"(
                Leaves only subgraph that are defined by new inputs and new outputs.

                Parameters
                ----------
                inputs : List[Place]
                    Array of new input places.

                outputs : List[Place]
                    Array of new output places.
            )");

    im.def("override_all_inputs",
           &ov::frontend::InputModel::override_all_inputs,
           py::arg("inputs"),
           R"(
                Modifies the graph to use new inputs instead of existing ones. New inputs
                should completely satisfy all existing outputs.

                Parameters
                ----------
                inputs : List[Place]
                    Array of new input places.
            )");

    im.def("override_all_outputs",
           &ov::frontend::InputModel::override_all_outputs,
           py::arg("outputs"),
           R"(
                Replaces all existing outputs with new ones removing all data flow that
                is not required for new outputs.

                Parameters
                ----------
                outputs : List[Place]
                    Vector with places that will become new outputs; may intersect existing outputs.
            )");

    im.def("set_element_type",
           &ov::frontend::InputModel::set_element_type,
           py::arg("place"),
           py::arg("type"),
           R"(
                Sets new element type for a place.

                Parameters
                ----------
                place : Place
                    Model place.

                type : ngraph.Type
                    New element type.
            )");

    im.def(
        "set_tensor_value",
        [](ov::frontend::InputModel& self, const ov::frontend::Place::Ptr& place, py::array& value) {
            // Convert to contiguous array if not already C-style.
            auto tensor = Common::tensor_from_numpy(value, false);
            self.set_tensor_value(place, (const void*)tensor.data());
        },
        py::arg("place"),
        py::arg("value"),
        R"(
            Sets new element type for a place.

            Parameters
            ----------
            place : Place
                Model place.

            value : ndarray
                New value to assign.
        )");
}
