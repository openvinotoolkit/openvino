// Copyright (C) 2018-2024 Intel Corporation
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
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/partial_shape.hpp"

namespace py = pybind11;

template <typename T>
bool compare_shape(const ov::PartialShape& a, const T& b) {
    if (a.is_dynamic()) {
        throw py::type_error("Cannot compare dynamic shape with " + std::string(py::str(py::type::of(b))));
    }
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), [](const ov::Dimension& elem_a, const py::handle& elem_b) {
               return elem_a == elem_b.cast<int64_t>();
           });
}

void regclass_graph_PartialShape(py::module m) {
    py::class_<ov::PartialShape, std::shared_ptr<ov::PartialShape>> shape(m, "PartialShape");
    shape.doc() = "openvino.runtime.PartialShape wraps ov::PartialShape";

    shape.def(py::init<const ov::Shape&>());
    shape.def(py::init<const ov::PartialShape&>());
    shape.def(py::init([](py::list& shape) {
        return Common::partial_shape_from_list(shape);
    }));
    shape.def(py::init([](py::tuple& shape) {
        return Common::partial_shape_from_list(shape.cast<py::list>());
    }));
    shape.def(py::init<const std::string&>(), py::arg("shape"));

    shape.def_static("dynamic",
                     &ov::PartialShape::dynamic,
                     py::arg("rank") = ov::Dimension(),
                     R"(
                       Construct a PartialShape with the given rank and all dimensions are dynamic.

                       :param rank: The rank of the PartialShape. This is the number of dimensions in the shape.
                       :type rank: openvino.Dimension
                       :return: A PartialShape with the given rank (or undefined rank if not provided), and all dimensions are dynamic.
                    )");

    shape.def_static(
        "dynamic",
        [](int64_t rank) {
            return ov::PartialShape::dynamic(ov::Dimension(rank));
        },
        py::arg("rank"),
        R"(
            Construct a PartialShape with the given rank and all dimensions are dynamic.

            :param rank: The rank of the PartialShape. This is the number of dimensions in the shape.
            :type rank: int
            :return: A PartialShape with the given rank, and all dimensions are dynamic.
        )");

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

                :param shape: The shape to be checked for compatibility with this shape.
                :type shape: openvino.runtime.PartialShape
                :return: True if this shape is compatible with s, else False.
                :rtype: bool
              )");
    shape.def("refines",
              &ov::PartialShape::refines,
              py::arg("shape"),
              R"(
                Check whether this shape is a refinement of the argument.

                :param shape: The shape which is being compared against this shape.
                :type shape: openvino.runtime.PartialShape
                :return: True if this shape refines s, else False.
                :rtype: bool
              )");
    shape.def("relaxes",
              &ov::PartialShape::relaxes,
              py::arg("shape"),
              R"(
                Check whether this shape is a relaxation of the argument.

                :param shape: The shape which is being compared against this shape.
                :type shape: openvino.runtime.PartialShape
                :return: True if this shape relaxes s, else False.
                :rtype: bool
              )");
    shape.def("same_scheme",
              &ov::PartialShape::same_scheme,
              py::arg("shape"),
              R"(
                Check whether this shape represents the same scheme as the argument.

                :param shape: The shape which is being compared against this shape.
                :type shape: openvino.runtime.PartialShape
                :return: True if shape represents the same scheme as s, else False.
                :rtype: bool
              )");
    shape.def("get_max_shape",
              &ov::PartialShape::get_max_shape,
              R"(
                :return: Get the max bounding shape.
                :rtype: openvino.runtime.Shape
              )");
    shape.def("get_min_shape",
              &ov::PartialShape::get_min_shape,
              R"(
                :return: Get the min bounding shape.
                :rtype: openvino.runtime.Shape
              )");
    shape.def("get_shape",
              &ov::PartialShape::get_shape,
              R"(
                :return: Get the unique shape.
                :rtype: openvino.runtime.Shape
              )");
    shape.def("to_shape",
              &ov::PartialShape::to_shape,
              R"(
                :return: Get the unique shape.
                :rtype: openvino.runtime.Shape
              )");
    shape.def(
        "get_dimension",
        [](const ov::PartialShape& self, size_t index) -> ov::Dimension {
            return self[index];
        },
        py::arg("index"),
        R"(
            Get the dimension at specified index of a partial shape.

            :param index: The index of dimension.
            :type index: int 
            :return: Get the particular dimension of a partial shape.
            :rtype: openvino.runtime.Dimension
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
    shape.def(
        "__eq__",
        [](const ov::PartialShape& a, const py::tuple& b) {
            return compare_shape<py::tuple>(a, b);
        },
        py::is_operator());

    shape.def(
        "__eq__",
        [](const ov::PartialShape& a, const py::list& b) {
            return compare_shape<py::list>(a, b);
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

    shape.def("__getitem__", [](const ov::PartialShape& self, int64_t key) {
        if (key < 0) {
            key += self.size();
        }
        return self[key];
    });

    shape.def("__getitem__", [](const ov::PartialShape& self, py::slice& slice) {
        size_t start, stop, step, slicelength;
        if (!slice.compute(self.size(), &start, &stop, &step, &slicelength)) {
            throw py::error_already_set();
        }
        ov::PartialShape result;
        result.resize(slicelength);
        Common::shape_helpers::get_slice(result, self, start, step, slicelength);
        return result;
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
        return "<" + Common::get_class_name(self) + ": " + py::cast(self).attr("__str__")().cast<std::string>() + ">";
    });

    shape.def("__copy__", [](const ov::PartialShape& self) -> ov::PartialShape {
        return ov::PartialShape(self);
    });

    shape.def(
        "__deepcopy__",
        [](const ov::PartialShape& self, py::dict) -> ov::PartialShape {
            return ov::PartialShape(self);
        },
        "memo");

    shape.def("to_string", &ov::PartialShape::to_string);
}
