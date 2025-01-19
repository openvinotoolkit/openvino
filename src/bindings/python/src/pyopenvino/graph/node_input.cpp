// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node_input.hpp"

#include <pybind11/stl.h>

#include "dict_attribute_visitor.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/node_input.hpp"

namespace py = pybind11;

using PyRTMap = ov::Node::RTMap;

PYBIND11_MAKE_OPAQUE(PyRTMap);

void regclass_graph_Input(py::module m) {
    py::class_<ov::Input<ov::Node>, std::shared_ptr<ov::Input<ov::Node>>> input(m, "Input", py::dynamic_attr());
    input.doc() = "openvino.Input wraps ov::Input<Node>";

    input.def("get_node",
              &ov::Input<ov::Node>::get_node,
              R"(
                Get node referenced by this input handle.

                :return: Node object referenced by this input handle.
                :rtype: openvino.Node
              )");
    input.def("get_index",
              &ov::Input<ov::Node>::get_index,
              R"(
                The index of the input referred to by this input handle.

                :return: Index value as integer.
                :rtype: int
              )");
    input.def("get_element_type",
              &ov::Input<ov::Node>::get_element_type,
              R"(
                The element type of the input referred to by this input handle.

                :return: Type of the input.
                :rtype: openvino.Type
              )");
    input.def("get_shape",
              &ov::Input<ov::Node>::get_shape,
              R"(
                The shape of the input referred to by this input handle.

                :return: Shape of the input.
                :rtype: openvino.Shape
              )");
    input.def("get_partial_shape",
              &ov::Input<ov::Node>::get_partial_shape,
              R"(
                The partial shape of the input referred to by this input handle.

                :return: PartialShape of the input.
                :rtype: openvino.PartialShape
              )");
    input.def("get_source_output",
              &ov::Input<ov::Node>::get_source_output,
              R"(
                A handle to the output that is connected to this input.

                :return: Output that is connected to the input.
                :rtype: openvino.Output
              )");
    input.def("get_tensor",
              &ov::Input<ov::Node>::get_tensor,
              py::return_value_policy::reference_internal,
              R"(
                A reference to the tensor descriptor for this input.

                :return: Tensor of the input.
                :rtype: openvino._pyopenvino.DescriptorTensor
               )");
    input.def("get_rt_info",
              (ov::RTMap & (ov::Input<ov::Node>::*)()) & ov::Input<ov::Node>::get_rt_info,
              py::return_value_policy::reference_internal,
              R"(
                Returns RTMap which is a dictionary of user defined runtime info.

                :return: A dictionary of user defined data.
                :rtype: openvino.RTMap
             )");
    input.def("replace_source_output",
              &ov::Input<ov::Node>::replace_source_output,
              py::arg("new_source_output"),
              R"(
                Replaces the source output of this input.

                :param new_source_output: A handle for the output that will replace this input's source.
                :type new_source_output: openvino.Input
              )");
    input.def_property_readonly("rt_info", (ov::RTMap & (ov::Input<ov::Node>::*)()) & ov::Input<ov::Node>::get_rt_info);
    input.def_property_readonly("rt_info",
                                (const ov::RTMap& (ov::Input<ov::Node>::*)() const) & ov::Input<ov::Node>::get_rt_info,
                                py::return_value_policy::reference_internal);

    input.def("__repr__", [](const ov::Input<ov::Node>& self) {
        std::stringstream shape_type_ss;
        shape_type_ss << " shape" << self.get_partial_shape() << " type: " << self.get_element_type();
        return "<" + Common::get_class_name(self) + ":" + shape_type_ss.str() + ">";
    });
}
