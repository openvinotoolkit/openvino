// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/frontend/place.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/manager.hpp"
#include "pyopenvino/graph/model.hpp"

namespace py = pybind11;

void regclass_frontend_Place(py::module m) {
    py::class_<ov::frontend::Place, std::shared_ptr<ov::frontend::Place>> place(m, "Place", py::dynamic_attr());
    place.doc() = "openvino.frontend.Place wraps ov::frontend::Place";

    place.def("is_input",
              &ov::frontend::Place::is_input,
              R"(
                Returns true if this place is input for a model.

                :return: True if this place is input for a model
                :rtype: bool
             )");

    place.def("is_output",
              &ov::frontend::Place::is_output,
              R"(
                Returns true if this place is output for a model.

                :return: True if this place is output for a model.
                :rtype: bool
             )");

    place.def("get_names",
              &ov::frontend::Place::get_names,
              R"(
                All associated names (synonyms) that identify this place in the graph in a framework specific way.

                :return: A vector of strings each representing a name that identifies this place in the graph.
                         Can be empty if there are no names associated with this place or name cannot be attached.
                :rtype: List[str]
             )");

    place.def("is_equal",
              &ov::frontend::Place::is_equal,
              py::arg("other"),
              R"(
                Returns true if another place is the same as this place.

                :param other: Another place object.
                :type other: openvino.frontend.Place
                :return: True if another place is the same as this place.
                :rtype: bool
             )");

    place.def("is_equal_data",
              &ov::frontend::Place::is_equal_data,
              py::arg("other"),
              R"(
                Returns true if another place points to the same data.
                Note: The same data means all places on path:
                      output port -> output edge -> tensor -> input edge -> input port.

                :param other: Another place object.
                :type other: openvino.frontend.Place
                :return: True if another place points to the same data.
                :rtype: bool
             )");

    place.def(
        "get_consuming_operations",
        [](const ov::frontend::Place& self, py::object outputName, py::object outputPortIndex) {
            if (outputName.is(py::none())) {
                if (outputPortIndex.is(py::none())) {
                    return self.get_consuming_operations();
                } else {
                    return self.get_consuming_operations(py::cast<int>(outputPortIndex));
                }
            } else {
                if (outputPortIndex.is(py::none())) {
                    return self.get_consuming_operations(py::cast<std::string>(outputName));
                } else {
                    return self.get_consuming_operations(py::cast<std::string>(outputName),
                                                         py::cast<int>(outputPortIndex));
                }
            }
        },
        py::arg("output_name") = py::none(),
        py::arg("output_port_index") = py::none(),
        R"(
                Returns references to all operation nodes that consume data from this place for specified output port.
                Note: It can be called for any kind of graph place searching for the first consuming operations.

                :param output_name: Name of output port group. May not be set if node has one output port group.
                :type output_name: str
                :param output_port_index: If place is an operational node it specifies which output port should be considered
                    May not be set if node has only one output port.
                :type output_port_index: int
                :return: A list with all operation node references that consumes data from this place
                :rtype: List[openvino.frontend.Place]
             )");

    place.def(
        "get_target_tensor",
        [](const ov::frontend::Place& self, py::object outputName, py::object outputPortIndex) {
            if (outputName.is(py::none())) {
                if (outputPortIndex.is(py::none())) {
                    return self.get_target_tensor();
                } else {
                    return self.get_target_tensor(py::cast<int>(outputPortIndex));
                }
            } else {
                if (outputPortIndex.is(py::none())) {
                    return self.get_target_tensor(py::cast<std::string>(outputName));
                } else {
                    return self.get_target_tensor(py::cast<std::string>(outputName), py::cast<int>(outputPortIndex));
                }
            }
        },
        py::arg("output_name") = py::none(),
        py::arg("output_port_index") = py::none(),
        R"(
                Returns a tensor place that gets data from this place; applicable for operations,
                output ports and output edges.

                :param output_name: Name of output port group. May not be set if node has one output port group.
                :type output_name: str
                :param output_port_index: Output port index if the current place is an operation node and has multiple output ports.
                    May not be set if place has only one output port.
                :type output_port_index: int
                :return: A tensor place which hold the resulting value for this place.
                :rtype: openvino.frontend.Place
             )");

    place.def(
        "get_producing_operation",
        [](const ov::frontend::Place& self, py::object inputName, py::object inputPortIndex) {
            if (inputName.is(py::none())) {
                if (inputPortIndex.is(py::none())) {
                    return self.get_producing_operation();
                } else {
                    return self.get_producing_operation(py::cast<int>(inputPortIndex));
                }
            } else {
                if (inputPortIndex.is(py::none())) {
                    return self.get_producing_operation(py::cast<std::string>(inputName));
                } else {
                    return self.get_producing_operation(py::cast<std::string>(inputName),
                                                        py::cast<int>(inputPortIndex));
                }
            }
        },
        py::arg("input_name") = py::none(),
        py::arg("input_port_index") = py::none(),
        R"(
                Get an operation node place that immediately produces data for this place.

                :param input_name: Name of port group. May not be set if node has one input port group.
                :type input_name: str
                :param input_port_index: If a given place is itself an operation node, this specifies a port index.
                    May not be set if place has only one input port.
                :type input_port_index: int
                :return: An operation place that produces data for this place.
                :rtype: openvino.frontend.Place
             )");

    place.def("get_producing_port",
              &ov::frontend::Place::get_producing_port,
              R"(
                Returns a port that produces data for this place.

                :return: A port place that produces data for this place.
                :rtype: openvino.frontend.Place
             )");

    place.def(
        "get_input_port",
        [](const ov::frontend::Place& self, py::object inputName, py::object inputPortIndex) {
            if (inputName.is(py::none())) {
                if (inputPortIndex.is(py::none())) {
                    return self.get_input_port();
                } else {
                    return self.get_input_port(py::cast<int>(inputPortIndex));
                }
            } else {
                if (inputPortIndex.is(py::none())) {
                    return self.get_input_port(py::cast<std::string>(inputName));
                } else {
                    return self.get_input_port(py::cast<std::string>(inputName), py::cast<int>(inputPortIndex));
                }
            }
        },
        py::arg("input_name") = py::none(),
        py::arg("input_port_index") = py::none(),
        R"(
                For operation node returns reference to an input port with specified name and index.

                :param input_name: Name of port group. May not be set if node has one input port group.
                :type input_name: str
                :param input_port_index: Input port index in a group. May not be set if node has one input port in a group.
                :type input_port_index: int
                :return: Appropriate input port place.
                :rtype: openvino.frontend.Place
             )");

    place.def(
        "get_output_port",
        [](const ov::frontend::Place& self, py::object outputName, py::object outputPortIndex) {
            if (outputName.is(py::none())) {
                if (outputPortIndex.is(py::none())) {
                    return self.get_output_port();
                } else {
                    return self.get_output_port(py::cast<int>(outputPortIndex));
                }
            } else {
                if (outputPortIndex.is(py::none())) {
                    return self.get_output_port(py::cast<std::string>(outputName));
                } else {
                    return self.get_output_port(py::cast<std::string>(outputName), py::cast<int>(outputPortIndex));
                }
            }
        },
        py::arg("output_name") = py::none(),
        py::arg("output_port_index") = py::none(),
        R"(
                For operation node returns reference to an output port with specified name and index.

                :param output_name: Name of output port group. May not be set if node has one output port group.
                :type output_name: str
                :param output_port_index: Output port index. May not be set if node has one output port in a group.
                :type output_port_index: int
                :return: Appropriate output port place.
                :rtype: openvino.frontend.Place
             )");

    place.def("get_consuming_ports",
              &ov::frontend::Place::get_consuming_ports,
              R"(
                Returns all input ports that consume data flows through this place.

                :return: Input ports that consume data flows through this place.
                :rtype: List[openvino.frontend.Place]
             )");

    place.def(
        "get_source_tensor",
        [](const ov::frontend::Place& self, py::object inputName, py::object inputPortIndex) {
            if (inputName.is(py::none())) {
                if (inputPortIndex.is(py::none())) {
                    return self.get_source_tensor();
                } else {
                    return self.get_source_tensor(py::cast<int>(inputPortIndex));
                }
            } else {
                if (inputPortIndex.is(py::none())) {
                    return self.get_source_tensor(py::cast<std::string>(inputName));
                } else {
                    return self.get_source_tensor(py::cast<std::string>(inputName), py::cast<int>(inputPortIndex));
                }
            }
        },
        py::arg("input_name") = py::none(),
        py::arg("input_port_index") = py::none(),
        R"(
                Returns a tensor place that supplies data for this place; applicable for operations,
                input ports and input edges.

                :param input_name : Name of port group. May not be set if node has one input port group.
                :type input_name: str
                :param input_port_index: Input port index for operational node. May not be specified if place has only one input port.
                :type input_port_index: int
                :return: A tensor place which supplies data for this place.
                :rtype: openvino.frontend.Place
             )");
}
