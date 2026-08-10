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

void regclass_graph_ConstOutputRTMap(py::module m) {
    auto warn = []() {
        Common::utils::deprecation_warning("Setting rt_info via ConstOutput",
                                           "2027.0",
                                           "Use a non-const output to modify runtime info.");
    };

    py::class_<ConstRTMapView, ov::RTMap>(m, "_ConstRTMapView")
        // mutating: warn then delegate to the real map
        .def("__setitem__",
             [warn](ConstRTMapView& self, const std::string& k, const std::string& v) {
                 warn();
                 (*self.actual)[k] = v;
             })
        .def("__setitem__",
             [warn](ConstRTMapView& self, const std::string& k, int64_t v) {
                 warn();
                 (*self.actual)[k] = v;
             })
        .def("__delitem__",
             [warn](ConstRTMapView& self, const std::string& k) {
                 warn();
                 auto it = self.actual->find(k);
                 if (it == self.actual->end())
                     throw py::key_error(k);
                 self.actual->erase(it);
             })
        // read-only: delegate without warning
        .def("__getitem__",
             [](ConstRTMapView& self, const std::string& k) -> py::object {
                 auto it = self.actual->find(k);
                 if (it == self.actual->end())
                     throw py::key_error(k);
                 return Common::utils::from_ov_any_no_leaves(it->second);
             })
        .def("__contains__",
             [](const ConstRTMapView& self, const std::string& k) {
                 return self.actual->count(k) > 0;
             })
        .def("__len__",
             [](const ConstRTMapView& self) {
                 return self.actual->size();
             })
        .def("__bool__",
             [](const ConstRTMapView& self) {
                 return !self.actual->empty();
             })
        .def(
            "__iter__",
            [](ConstRTMapView& self) {
                return py::make_key_iterator(self.actual->begin(), self.actual->end());
            },
            py::keep_alive<0, 1>())
        .def(
            "keys",
            [](ConstRTMapView& self) {
                return py::make_key_iterator(self.actual->begin(), self.actual->end());
            },
            py::keep_alive<0, 1>())
        .def(
            "items",
            [](ConstRTMapView& self) -> py::object {
                return py::cast(*self.actual, py::return_value_policy::reference).attr("items")();
            },
            py::keep_alive<0, 1>())
        .def(
            "values",
            [](ConstRTMapView& self) -> py::object {
                return py::cast(*self.actual, py::return_value_policy::reference).attr("values")();
            },
            py::keep_alive<0, 1>())
        .def("__repr__", [](const ConstRTMapView& self) {
            return std::string("<RTMap>");
        });
}

template <typename T>
void def_type_dependent_functions(py::class_<ov::Output<T>, std::shared_ptr<ov::Output<T>>>& output) {}

template <>
void def_type_dependent_functions<const ov::Node>(
    py::class_<ov::Output<const ov::Node>, std::shared_ptr<ov::Output<const ov::Node>>>& output) {
    auto getter = [](py::object self_obj) -> py::object {
        ov::RTMap& rt = const_cast<ov::RTMap&>(self_obj.cast<ov::Output<const ov::Node>&>().get_rt_info());
        return py::cast(std::make_unique<ConstRTMapView>(rt, self_obj));
    };
    output.def("get_rt_info",
               getter,
               R"(
            Returns a view of the RTMap for this output.
            Reads are transparent; writes are deprecated (to be removed in 2027.0).
            Use a non-const output to modify runtime info.

            :return: View of runtime info dictionary.
            :rtype: openvino.RTMap
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
