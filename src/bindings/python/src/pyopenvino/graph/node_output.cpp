// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node_output.hpp"

#include <pybind11/stl.h>

#include "dict_attribute_visitor.hpp"
#include "pyopenvino/graph/node_output.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

template void regclass_graph_Output<ov::Node>(py::module m, std::string typestring);
template void regclass_graph_Output<const ov::Node>(py::module m, std::string typestring);

template <typename T>
void def_type_dependent_functions(py::class_<ov::Output<T>, std::shared_ptr<ov::Output<T>>>& output) {}

template <>
void def_type_dependent_functions<const ov::Node>(
    py::class_<ov::Output<const ov::Node>, std::shared_ptr<ov::Output<const ov::Node>>>& output) {
    // def_property_readonly does not support keep_alive, so
    // self_obj is passed to _ConstOutputRTMap and is stored as the owner,
    // keeping the node alive for as long as the proxy object exists.
    auto getter = [](py::object self_obj) -> py::object {
        auto& self = self_obj.cast<ov::Output<const ov::Node>&>();
        ov::RTMap& rt = const_cast<ov::RTMap&>(self.get_rt_info());
        py::object py_rtmap = py::cast(rt, py::return_value_policy::reference);
        return py::module_::import("openvino._ov_api").attr("_ConstOutputRTMap")(py_rtmap, self_obj);
    };
    output.def("get_rt_info",
               getter,
               R"(
            Returns a view of the RTMap for this output.
            Writes to this RTMap object emit a DeprecationWarning;
            use a non-const output (e.g. node.output(i)) to modify runtime info.

            :return: View of runtime info dictionary.
            :rtype: openvino._ov_api._ConstOutputRTMap
        )");
    output.def_property_readonly("rt_info", getter);
}

template <>
void def_type_dependent_functions<ov::Node>(
    py::class_<ov::Output<ov::Node>, std::shared_ptr<ov::Output<ov::Node>>>& output) {
    output.def("get_rt_info",
               (ov::RTMap & (ov::Output<ov::Node>::*)()) & ov::Output<ov::Node>::get_rt_info,
               py::return_value_policy::reference_internal,
               R"(
            Returns RTMap which is a dictionary of user defined runtime info.

            :return: A dictionary of user defined data.
            :rtype: openvino.RTMap
        )");
    output.def_property_readonly("rt_info",
                                 (ov::RTMap & (ov::Output<ov::Node>::*)()) & ov::Output<ov::Node>::get_rt_info,
                                 py::return_value_policy::reference_internal);
    output.def("set_names",
               &ov::Output<ov::Node>::set_names,
               py::arg("names"),
               R"(
            Set tensor names associated with this output.

            :param names: set of tensor names.
            :type names: set[str]
            )");
    output.def("add_names",
               &ov::Output<ov::Node>::add_names,
               py::arg("names"),
               R"(
            Add tensor names associated with this output.

            :param names: set of tensor names.
            :type names: set[str]
            )");
    output.def("remove_target_input",
               &ov::Output<ov::Node>::remove_target_input,
               py::arg("target_input"),
               R"(
                Removes a target input from the output referenced by this output handle.

                :param target_input: The target input to remove.
                :type target_input: openvino.Output
               )");
    output.def("replace",
               &ov::Output<ov::Node>::replace,
               py::arg("replacement"),
               R"(
                Replace all users of this value with replacement.

                :param replacement: The node that is a replacement.
                :type replacement: openvino.Output
               )");
    output.def(
        "set_rt_info",
        [](ov::Output<ov::Node>& self, const py::object& value, const py::str& key) -> void {
            self.get_rt_info()[key.cast<std::string>()] = Common::utils::py_object_to_any(value);
        },
        py::arg("value"),
        py::arg("key"),
        R"(
                Add a value to the runtime info.

                :param value: Value for the runtime info.
                :type value: Any
                :param key: String that defines a key in the runtime info dictionary.
                :type key: str
             )");
}
