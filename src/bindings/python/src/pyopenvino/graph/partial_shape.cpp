// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/partial_shape.hpp"  // ov::PartialShape

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "openvino/core/dimension.hpp"  // ov::Dimension
#include "openvino/core/shape.hpp"      // ov::Shape
#include "pyopenvino/graph/partial_shape.hpp"

namespace py = pybind11;

static const char* CAPSULE_NAME = "ngraph_partial_shape";

void regclass_graph_PartialShape(py::module m) {
    py::class_<ov::PartialShape, std::shared_ptr<ov::PartialShape>> shape(m, "PartialShape");
    shape.doc() = "openvino.runtime.PartialShape wraps ov::PartialShape";

    shape.def(py::init([](const std::vector<int64_t>& dimensions) {
        return ov::PartialShape(std::vector<ov::Dimension>(dimensions.begin(), dimensions.end()));
    }));
    shape.def(py::init<const std::initializer_list<size_t>&>());
    shape.def(py::init<const std::vector<size_t>&>());
    shape.def(py::init<const std::initializer_list<ov::Dimension>&>());
    shape.def(py::init<const std::vector<ov::Dimension>&>());
    shape.def(py::init<const ov::Shape&>());
    shape.def(py::init<const ov::PartialShape&>());

    shape.def_static("dynamic", &ov::PartialShape::dynamic, py::arg("rank") = ov::Dimension());

    shape.def_property_readonly("is_dynamic",
                                &ov::PartialShape::is_dynamic,
                                R"(
                                    False if this shape is static, else True.
                                    A shape is considered static if it has static rank,
                                    and all dimensions of the shape are static.
                                )");
    shape.def_property_readonly("is_static",
                                &ov::PartialShape::is_static,
                                R"(
                                    True if this shape is static, else False.
                                    A shape is considered static if it has static rank,
                                    and all dimensions of the shape are static.
                                )");
    shape.def_property_readonly("rank",
                                &ov::PartialShape::rank,
                                R"(
                                    The rank of the shape.
                                )");
    shape.def_property_readonly("all_non_negative",
                                &ov::PartialShape::all_non_negative,
                                R"(
                                    True if all static dimensions of the tensor are
                                    non-negative, else False.
                                )");

    shape.def("compatible",
              &ov::PartialShape::compatible,
              py::arg("shape"),
              R"(
                Check whether this shape is compatible with the argument, i.e.,
                whether it is possible to merge them.

                Parameters
                ----------
                shape : PartialShape
                    The shape to be checked for compatibility with this shape.


                Returns
                ----------
                compatible : bool
                    True if this shape is compatible with s, else False.
              )");
    shape.def("refines",
              &ov::PartialShape::refines,
              py::arg("shape"),
              R"(
                Check whether this shape is a refinement of the argument.

                Parameters
                ----------
                shape : PartialShape
                    The shape which is being compared against this shape.

                Returns
                ----------
                refines : bool
                    True if this shape refines s, else False.
              )");
    shape.def("relaxes",
              &ov::PartialShape::relaxes,
              py::arg("shape"),
              R"(
                Check whether this shape is a relaxation of the argument.

                Parameters
                ----------
                shape : PartialShape
                    The shape which is being compared against this shape.

                Returns
                ----------
                relaxes : bool
                    True if this shape relaxes s, else False.
              )");
    shape.def("same_scheme",
              &ov::PartialShape::same_scheme,
              py::arg("shape"),
              R"(
                Check whether this shape represents the same scheme as the argument.

                Parameters
                ----------
                shape : PartialShape
                    The shape which is being compared against this shape.

                Returns
                ----------
                same_scheme : bool
                    True if shape represents the same scheme as s, else False.
              )");
    shape.def("get_max_shape",
              &ov::PartialShape::get_max_shape,
              R"(
                Returns
                ----------
                get_max_shape : Shape
                    Get the max bounding shape.
              )");
    shape.def("get_min_shape",
              &ov::PartialShape::get_min_shape,
              R"(
                Returns
                ----------
                get_min_shape : Shape
                    Get the min bounding shape.
              )");
    shape.def("get_shape",
              &ov::PartialShape::get_shape,
              R"(
                Returns
                ----------
                get_shape : Shape
                    Get the unique shape.
              )");
    shape.def("to_shape",
              &ov::PartialShape::to_shape,
              R"(
                Returns
                ----------
                to_shapess : Shape
                    Get the unique shape.
              )");
    shape.def(
        "get_dimension",
        [](const ov::PartialShape& self, size_t index) -> ov::Dimension {
            return self[index];
        },
        py::arg("index"),
        R"(
                Get the dimension at specified index of a partial shape.

                Parameters
                ----------
                index : int
                    The index of dimension

                Returns
                ----------
                get_dimension : Dimension
                    Get the particular dimension of a partial shape.
              )");

    shape.def(
        "__eq__",
        [](const ov::PartialShape& a, const ov::PartialShape& b) {
            return a == b;
        },
        py::is_operator());
    shape.def(
        "__eq__",
        [](const ov::PartialShape& a, const ov::Shape& b) {
            return a == b;
        },
        py::is_operator());

    shape.def("__len__", [](const ov::PartialShape& self) {
        return self.size();
    });

    shape.def("__setitem__", [](ov::PartialShape& self, size_t key, ov::Dimension::value_type d) {
        self[key] = d;
    });

    shape.def("__setitem__", [](ov::PartialShape& self, size_t key, ov::Dimension& d) {
        self[key] = d;
    });

    shape.def("__getitem__", [](const ov::PartialShape& self, size_t key) {
        return self[key];
    });

    shape.def(
        "__iter__",
        [](ov::PartialShape& self) {
            return py::make_iterator(self.begin(), self.end());
        },
        py::keep_alive<0, 1>()); /* Keep vector alive while iterator is used */

    shape.def("__str__", [](const ov::PartialShape& self) -> std::string {
        std::stringstream ss;
        ss << self;
        return ss.str();
    });

    shape.def("__repr__", [](const ov::PartialShape& self) -> std::string {
        return "<PartialShape: " + py::cast(self).attr("__str__")().cast<std::string>() + ">";
    });

    shape.def_static("from_capsule", [](py::object* capsule) {
        // get the underlying PyObject* which is a PyCapsule pointer
        auto* pybind_capsule_ptr = capsule->ptr();
        // extract the pointer stored in the PyCapsule under the name CAPSULE_NAME
        auto* capsule_ptr = PyCapsule_GetPointer(pybind_capsule_ptr, CAPSULE_NAME);

        auto* ngraph_pShape = static_cast<std::shared_ptr<ov::PartialShape>*>(capsule_ptr);
        if (ngraph_pShape && *ngraph_pShape) {
            return *ngraph_pShape;
        } else {
            throw std::runtime_error("The provided capsule does not contain an ov::PartialShape");
        }
    });
}
