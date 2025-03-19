// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/frontend/input_model.hpp"

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

                :param tensor_name: Name of tensor.
                :type tensor_name: str
                :return: Tensor place corresponding to specified tensor name.
                :rtype: openvino.frontend.Place
             )");

    im.def("get_place_by_input_index",
           &ov::frontend::InputModel::get_place_by_input_index,
           py::arg("input_idx"),
           R"(
                Returns a tensor place by an input index.

                :param input_idx: Index of model input.
                :type input_idx: int
                :return: Tensor place corresponding to specified input index or nullptr.
                :rtype: openvino.frontend.Place
             )");

    im.def("get_place_by_operation_name",
           &ov::frontend::InputModel::get_place_by_operation_name,
           py::arg("operation_name"),
           R"(
                Returns an operation place by an operation name following framework conventions, or
                nullptr if an operation with this name doesn't exist.

                :param operation_name: Name of operation.
                :type operation_name: str
                :return: Place representing operation.
                :rtype: openvino.frontend.Place
             )");

    im.def("get_place_by_operation_name_and_input_port",
           &ov::frontend::InputModel::get_place_by_operation_name_and_input_port,
           py::arg("operation_name"),
           py::arg("input_port_index"),
           R"(
                Returns an input port place by operation name and appropriate port index.

                :param operation_name: Name of operation.
                :type operation_name: str
                :param input_port_index: Index of input port for this operation.
                :type input_port_index: int
                :return: Place representing input port of operation.
                :rtype: openvino.frontend.Place
             )");

    im.def("get_place_by_operation_name_and_output_port",
           &ov::frontend::InputModel::get_place_by_operation_name_and_output_port,
           py::arg("operation_name"),
           py::arg("output_port_index"),
           R"(
                Returns an output port place by operation name and appropriate port index.

                :param operation_name: Name of operation.
                :type operation_name: str
                :param output_port_index: Index of output port for this operation.
                :type output_port_index: int
                :return: Place representing output port of operation.
                :rtype: openvino.frontend.Place
             )");

    im.def("set_name_for_tensor",
           &ov::frontend::InputModel::set_name_for_tensor,
           py::arg("tensor"),
           py::arg("new_name"),
           R"(
                Sets name for tensor. Overwrites existing names of this place.

                :param tensor: Tensor place.
                :type tensor: openvino.frontend.Place
                :param new_name: New name for this tensor.
                :type new_name: str
            )");

    im.def("add_name_for_tensor",
           &ov::frontend::InputModel::add_name_for_tensor,
           py::arg("tensor"),
           py::arg("new_name"),
           R"(
                Adds new name for tensor

                :param tensor: Tensor place.
                :type tensor: openvino.frontend.Place
                :param new_name: New name to be added to this place.
                :type new_name: str
            )");

    im.def("set_name_for_operation",
           &ov::frontend::InputModel::set_name_for_operation,
           py::arg("operation"),
           py::arg("new_name"),
           R"(
                Adds new name for tensor.

                :param operation: Operation place.
                :type operation: openvino.frontend.Place
                :param new_name: New name for this operation.
                :type new_name: str
            )");

    im.def("free_name_for_tensor",
           &ov::frontend::InputModel::free_name_for_tensor,
           py::arg("name"),
           R"(
                Unassign specified name from tensor place(s).

                :param name: Name of tensor.
                :type name: str
            )");

    im.def("free_name_for_operation",
           &ov::frontend::InputModel::free_name_for_operation,
           py::arg("name"),
           R"(
                Unassign specified name from operation place(s).

                :param name: Name of operation.
                :type name: str
            )");

    im.def("set_name_for_dimension",
           &ov::frontend::InputModel::set_name_for_dimension,
           py::arg("place"),
           py::arg("dim_index"),
           py::arg("dim_name"),
           R"(
                Set name for a particular dimension of a place (e.g. batch dimension).

                :param place: Model's place.
                :type place: openvino.frontend.Place
                :param dim_index: Dimension index.
                :type dim_index: int
                :param dim_name: Name to assign on this dimension.
                :type dum_name: str
            )");

    im.def("cut_and_add_new_input",
           &ov::frontend::InputModel::cut_and_add_new_input,
           py::arg("place"),
           py::arg("new_name") = std::string(),
           R"(
                Cut immediately before this place and assign this place as new input; prune
                all nodes that don't contribute to any output.

               :param place: New place to be assigned as input.
               :type place: openvino.frontend.Place
               :param new_name: Optional new name assigned to this input place.
               :type new_name: str
            )");

    im.def("cut_and_add_new_output",
           &ov::frontend::InputModel::cut_and_add_new_output,
           py::arg("place"),
           py::arg("new_name") = std::string(),
           R"(
                Cut immediately before this place and assign this place as new output; prune
                all nodes that don't contribute to any output.

                :param place: New place to be assigned as output.
                :type place: openvino.frontend.Place
                :param new_name: Optional new name assigned to this output place.
                :type new_name: str
            )");

    im.def("add_output",
           &ov::frontend::InputModel::add_output,
           py::arg("place"),
           R"(
                Assign this place as new output or add necessary nodes to represent a new output.

                :param place: Anchor point to add an output.
                :type place: openvino.frontend.Place
            )");

    im.def("remove_output",
           &ov::frontend::InputModel::remove_output,
           py::arg("place"),
           R"(
                Removes any sinks directly attached to this place with all inbound data flow
                if it is not required by any other output.

                :param place: Model place.
                :type place: openvino.frontend.Place
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

                :param place: Model place.
                :type place: openvino.frontend.Place
                :param shape: Partial shape for this place.
                :type shape: openvino.PartialShape
            )");

    im.def("get_partial_shape",
           &ov::frontend::InputModel::get_partial_shape,
           py::arg("place"),
           R"(
                Returns current partial shape used for this place.

                :param place: Model place.
                :type place: openvino.frontend.Place
                :return: Partial shape for this place.
                :rtype: openvino.PartialShape
            )");

    im.def("get_inputs",
           &ov::frontend::InputModel::get_inputs,
           R"(
                Returns all inputs for a model.

                :return: A list of input places.
                :rtype: List[openvino.frontend.Place]
            )");

    im.def("get_outputs",
           &ov::frontend::InputModel::get_outputs,
           R"(
                Returns all outputs for a model. An output is a terminal place in a graph where data escapes the flow.

                :return: A list of output places.
                :rtype: List[openvino.frontend.Place]
            )");

    im.def("extract_subgraph",
           &ov::frontend::InputModel::extract_subgraph,
           py::arg("inputs"),
           py::arg("outputs"),
           R"(
                Leaves only subgraph that are defined by new inputs and new outputs.

                :param inputs: Array of new input places.
                :type inputs: List[openvino.frontend.Place]
                :param outputs: Array of new output places.
                :type outputs: List[openvino.frontend.Place]
            )");

    im.def("override_all_inputs",
           &ov::frontend::InputModel::override_all_inputs,
           py::arg("inputs"),
           R"(
                Modifies the graph to use new inputs instead of existing ones. New inputs
                should completely satisfy all existing outputs.

                :param inputs: Array of new input places.
                :type inputs: List[openvino.frontend.Place]
            )");

    im.def("override_all_outputs",
           &ov::frontend::InputModel::override_all_outputs,
           py::arg("outputs"),
           R"(
                Replaces all existing outputs with new ones removing all data flow that
                is not required for new outputs.

                :param outputs: Vector with places that will become new outputs; may intersect existing outputs.
                :type outputs: List[openvino.frontend.Place]
            )");

    im.def("set_element_type",
           &ov::frontend::InputModel::set_element_type,
           py::arg("place"),
           py::arg("type"),
           R"(
                Sets new element type for a place.

                :param place: Model place.
                :type place: openvino.frontend.Place
                :param type: New element type.
                :type type: openvino.Type
            )");

    im.def("get_element_type",
           &ov::frontend::InputModel::get_element_type,
           py::arg("place"),
           R"(
                Returns current element type used for this place.

                :param place: Model place.
                :type place: openvino.frontend.Place
                :return: Element type for this place.
                :rtype: openvino.Type
            )");

    im.def(
        "set_tensor_value",
        [](ov::frontend::InputModel& self, const ov::frontend::Place::Ptr& place, py::array& value) {
            // Convert to contiguous array if not already C-style.
            auto tensor = Common::object_from_data<ov::Tensor>(value, false);
            self.set_tensor_value(place, (const void*)tensor.data());
        },
        py::arg("place"),
        py::arg("value"),
        R"(
            Sets new element type for a place.

            :param place: Model place.
            :type place: openvino.frontend.Place
            :param value: New value to assign.
            :type value: numpy.ndarray
        )");
}
