// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/tensor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "openvino/runtime/tensor.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/core/remote_tensor.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

void regclass_Tensor(py::module m) {
    py::class_<ov::Tensor, std::shared_ptr<ov::Tensor>> cls(m, "Tensor");
    cls.doc() = "openvino.runtime.Tensor holding either copy of memory or shared host memory.";

    cls.def(py::init([](py::array& array, bool shared_memory) {
                return Common::object_from_data<ov::Tensor>(array, shared_memory);
            }),
            py::arg("array"),
            py::arg("shared_memory") = false,
            py::ov_extension::conditional_keep_alive<1, 2, 3>(),
            R"(
                Tensor's special constructor.

                :param array: Array to create the tensor from.
                :type array: numpy.array
                :param shared_memory: If `True`, this Tensor memory is being shared with a host.
                                      Any action performed on the host memory is reflected on this Tensor's memory!
                                      If `False`, data is being copied to this Tensor.
                                      Requires data to be C_CONTIGUOUS if `True`.
                                      If the passed array contains strings, the flag must be set to `False'.
                :type shared_memory: bool
            )");

    cls.def(py::init([](py::array& array, const ov::Shape& shape, const ov::element::Type& ov_type) {
                return Common::tensor_from_pointer(array, shape, ov_type);
            }),
            py::arg("array"),
            py::arg("shape"),
            py::arg("type") = ov::element::undefined,
            py::keep_alive<1, 2>(),
            R"(
                Another Tensor's special constructor.

                Represents array in the memory with given shape and element type.
                It's recommended to use this constructor only for wrapping array's
                memory with the specific openvino element type parameter.

                :param array: C_CONTIGUOUS numpy array which will be wrapped in
                              openvino.runtime.Tensor with given parameters (shape
                              and element_type). Array's memory is being shared with a host.
                              Any action performed on the host memory will be reflected on this Tensor's memory!
                :type array: numpy.array
                :param shape: Shape of the new tensor.
                :type shape: openvino.runtime.Shape
                :param type: Element type
                :type type: openvino.runtime.Type

                :Example:
                .. code-block:: python

                    import openvino.runtime as ov
                    import numpy as np

                    arr = np.array(shape=(100), dtype=np.uint8)
                    t = ov.Tensor(arr, ov.Shape([100, 8]), ov.Type.u1)
            )");

    cls.def(py::init([](py::array& array, const std::vector<size_t> shape, const ov::element::Type& ov_type) {
                return Common::tensor_from_pointer(array, shape, ov_type);
            }),
            py::arg("array"),
            py::arg("shape"),
            py::arg("type") = ov::element::undefined,
            py::keep_alive<1, 2>(),
            R"(
                 Another Tensor's special constructor.

                Represents array in the memory with given shape and element type.
                It's recommended to use this constructor only for wrapping array's
                memory with the specific openvino element type parameter.

                :param array: C_CONTIGUOUS numpy array which will be wrapped in
                              openvino.runtime.Tensor with given parameters (shape
                              and element_type). Array's memory is being shared with a host.
                              Any action performed on the host memory will be reflected on this Tensor's memory!
                :type array: numpy.array
                :param shape: Shape of the new tensor.
                :type shape: list or tuple
                :param type: Element type.
                :type type: openvino.runtime.Type

                :Example:
                .. code-block:: python

                    import openvino.runtime as ov
                    import numpy as np

                    arr = np.array(shape=(100), dtype=np.uint8)
                    t = ov.Tensor(arr, [100, 8], ov.Type.u1)
            )");

    // It may clash in future with overloads like <ov::Coordinate, ov::Coordinate>
    cls.def(py::init([](py::list& list) {
                auto array = py::array(list);
                return Common::object_from_data<ov::Tensor>(array, false);
            }),
            py::arg("list"),
            R"(
                Tensor's special constructor.

                Creates a Tensor from a given Python list.
                Warning: It is always a copy of list's data!

                :param array: List to create the tensor from.
                :type array: List[int, float, str]
            )");

    cls.def(py::init<const ov::element::Type, const ov::Shape>(), py::arg("type"), py::arg("shape"));

    cls.def(py::init<const ov::element::Type, const std::vector<size_t>>(), py::arg("type"), py::arg("shape"));

    cls.def(py::init([](py::dtype& np_dtype, std::vector<size_t>& shape) {
                return ov::Tensor(Common::type_helpers::get_ov_type(np_dtype), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::object& np_literal, std::vector<size_t>& shape) {
                auto dtype = py::dtype::from_args(np_literal);
                return ov::Tensor(Common::type_helpers::get_ov_type(dtype), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::dtype& np_dtype, const ov::Shape& shape) {
                return ov::Tensor(Common::type_helpers::get_ov_type(np_dtype), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::object& np_literal, const ov::Shape& shape) {
                auto dtype = py::dtype::from_args(np_literal);
                return ov::Tensor(Common::type_helpers::get_ov_type(dtype), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init<ov::Output<ov::Node>>(),
            py::arg("port"),
            R"(
                Constructs Tensor using port from node.
                Type and shape will be taken from the port.

                :param port: Output port from a node.
                :type param: openvino.runtime.Output
             )");

    cls.def(py::init([](ov::Output<ov::Node>& port, py::array& array) {
                return Common::tensor_from_pointer(array, port);
            }),
            py::arg("port"),
            py::arg("array"),
            py::keep_alive<1, 3>(),
            R"(
                Constructs Tensor using port from node.
                Type and shape will be taken from the port.

                :param port: Output port from a node.
                :type param: openvino.runtime.Output
                :param array: C_CONTIGUOUS numpy array which will be wrapped in
                              openvino.runtime.Tensor. Array's memory is being shared wi a host.
                              Any action performed on the host memory will be reflected on this Tensor's memory!
                :type array: numpy.array
             )");

    cls.def(py::init<const ov::Output<const ov::Node>>(),
            py::arg("port"),
            R"(
            Constructs Tensor using port from node.
            Type and shape will be taken from the port.

            :param port: Output port from a node.
            :type param: openvino.runtime.ConstOutput
            )");

    cls.def(py::init([](const ov::Output<const ov::Node>& port, py::array& array) {
                return Common::tensor_from_pointer(array, port);
            }),
            py::arg("port"),
            py::arg("array"),
            py::keep_alive<1, 3>(),
            R"(
                Constructs Tensor using port from node.
                Type and shape will be taken from the port.

                :param port: Output port from a node.
                :type param: openvino.runtime.ConstOutput
                :param array: C_CONTIGUOUS numpy array which will be wrapped in
                              openvino.runtime.Tensor. Array's memory is being shared with a host.
                              Any action performed on the host memory will be reflected on this Tensor's memory!
                :type array: numpy.array
             )");

    cls.def(py::init<ov::Tensor, ov::Coordinate, ov::Coordinate>(), py::arg("other"), py::arg("begin"), py::arg("end"));

    cls.def(py::init<ov::Tensor, std::vector<size_t>, std::vector<size_t>>(),
            py::arg("other"),
            py::arg("begin"),
            py::arg("end"));

    cls.def("get_element_type",
            &ov::Tensor::get_element_type,
            R"(
            Gets Tensor's element type.

            :rtype: openvino.runtime.Type
            )");

    cls.def_property_readonly("element_type",
                              &ov::Tensor::get_element_type,
                              R"(
                                Tensor's element type.

                                :rtype: openvino.runtime.Type
                              )");

    cls.def("get_size",
            &ov::Tensor::get_size,
            R"(
            Gets Tensor's size as total number of elements.

            :rtype: int
            )");

    cls.def_property_readonly("size",
                              &ov::Tensor::get_size,
                              R"(
                                Tensor's size as total number of elements.

                                :rtype: int
                              )");

    cls.def("get_byte_size",
            &ov::Tensor::get_byte_size,
            R"(
            Gets Tensor's size in bytes.

            :rtype: int
            )");

    cls.def_property_readonly("byte_size",
                              &ov::Tensor::get_byte_size,
                              R"(
                                Tensor's size in bytes.

                                :rtype: int
                              )");

    cls.def("get_strides",
            &ov::Tensor::get_strides,
            R"(
            Gets Tensor's strides in bytes.

            :rtype: openvino.runtime.Strides
            )");

    cls.def_property_readonly("strides",
                              &ov::Tensor::get_strides,
                              R"(
                                Tensor's strides in bytes.

                                :rtype: openvino.runtime.Strides
                              )");

    cls.def_property_readonly(
        "data",
        [](ov::Tensor& self) {
            return Common::array_helpers::array_from_tensor(std::forward<ov::Tensor>(self), true);
        },
        R"(
            Access to Tensor's data.

            Returns numpy array with corresponding shape and dtype.

            For tensors with OpenVINO specific element type, such as u1, u4 or i4
            it returns linear array, with uint8 / int8 numpy dtype.

            For tensors with string element type, returns a numpy array of bytes
            without any decoding.
            To change the underlaying data use `str_data`/`bytes_data` properties
            or the `copy_from` function.
            Warning: Data of string type is always a copy of underlaying memory!

            :rtype: numpy.array
        )");

    cls.def_property(
        "bytes_data",
        [](ov::Tensor& self) {
            return Common::string_helpers::bytes_array_from_tensor(std::forward<ov::Tensor>(self));
        },
        [](ov::Tensor& self, py::object& other) {
            if (py::isinstance<py::array>(other)) {
                auto array = other.cast<py::array>();
                Common::string_helpers::fill_string_tensor_data(self, array);
            } else if (py::isinstance<py::list>(other)) {
                auto array = py::array(other.cast<py::list>());
                Common::string_helpers::fill_string_tensor_data(self, array);
            } else {
                OPENVINO_THROW("Invalid data to fill String Tensor!");
            }
            return;
        },
        R"(
            Access to Tensor's data with string Type in `np.bytes_` dtype.

            Getter returns a numpy array with corresponding shape and dtype.
            Warning: Data of string type is always a copy of underlaying memory!

            Setter fills underlaying Tensor's memory by copying strings from `other`.
            `other` must have the same size (number of elements) as the Tensor.
            Tensor's shape is not changed by performing this operation!
        )");

    cls.def_property(
        "str_data",
        [](ov::Tensor& self) {
            return Common::string_helpers::string_array_from_tensor(std::forward<ov::Tensor>(self));
        },
        [](ov::Tensor& self, py::object& other) {
            if (py::isinstance<py::array>(other)) {
                auto array = other.cast<py::array>();
                Common::string_helpers::fill_string_tensor_data(self, array);
            } else if (py::isinstance<py::list>(other)) {
                auto array = py::array(other.cast<py::list>());
                Common::string_helpers::fill_string_tensor_data(self, array);
            } else {
                OPENVINO_THROW("Invalid data to fill String Tensor!");
            }
            return;
        },
        R"(
            Access to Tensor's data with string Type in `np.str_` dtype.

            Getter returns a numpy array with corresponding shape and dtype.
            Warning: Data of string type is always a copy of underlaying memory!

            Setter fills underlaying Tensor's memory by copying strings from `other`.
            `other` must have the same size (number of elements) as the Tensor.
            Tensor's shape is not changed by performing this operation!
        )");

    cls.def("get_shape",
            &ov::Tensor::get_shape,
            R"(
            Gets Tensor's shape.

            :rtype: openvino.runtime.Shape
            )");

    cls.def("set_shape",
            &ov::Tensor::set_shape,
            R"(
            Sets Tensor's shape.
            )");

    cls.def(
        "set_shape",
        [](ov::Tensor& self, std::vector<size_t>& shape) {
            self.set_shape(shape);
        },
        R"(
            Sets Tensor's shape.
        )");

    cls.def(
        "copy_to",
        [](ov::Tensor& self, ov::Tensor& dst) {
            return self.copy_to(dst);
        },
        py::arg("target_tensor"),
        R"(
        Copy tensor's data to a destination tensor. The destination tensor should have the same element type and shape.

        :param target_tensor: The destination tensor to which the data will be copied.
        :type target_tensor: openvino.Tensor
    )");

    cls.def(
        "copy_to",
        [](ov::Tensor& self, RemoteTensorWrapper& dst) {
            return self.copy_to(dst.tensor);
        },
        py::arg("target_tensor"),
        R"(
        Copy tensor's data to a destination remote tensor. The destination remote tensor should have the same element type.
        In case of RoiRemoteTensor, the destination tensor should also have the same shape.

        :param target_tensor: The destination remote tensor to which the data will be copied.
        :type target_tensor: openvino.RemoteTensor
    )");

    cls.def(
        "copy_from",
        [](ov::Tensor& self, ov::Tensor& source) {
            return source.copy_to(self);
        },
        py::arg("source_tensor"),
        R"(
        Copy source tensor's data to this tensor. Tensors should have the same element type and shape.

        :param source_tensor: The source tensor from which the data will be copied.
        :type source_tensor: openvino.Tensor
    )");

    cls.def(
        "copy_from",
        [](ov::Tensor& self, RemoteTensorWrapper& source) {
            return source.tensor.copy_to(self);
        },
        py::arg("source_tensor"),
        R"(
        Copy source remote tensor's data to this tensor. Tensors should have the same element type.
        In case of RoiTensor, tensors should also have the same shape.

        :param source_tensor: The source remote tensor from which the data will be copied.
        :type source_tensor: openvino.RemoteTensor
    )");

    cls.def(
        "copy_from",
        [](ov::Tensor& self, py::array& source) {
            auto _source = Common::object_from_data<ov::Tensor>(source, false);
            if (self.get_shape() != _source.get_shape()) {
                self.set_shape(_source.get_shape());
            }
            return _source.copy_to(self);
        },
        py::arg("source"),
        R"(
        Copy the source to this tensor. This tensor and the source should have the same element type.
        Shape will be adjusted if there is a mismatch.
    )");

    cls.def(
        "copy_from",
        [](ov::Tensor& self, py::list& source) {
            auto array = py::array(source);
            auto _source = Common::object_from_data<ov::Tensor>(array, false);
            if (self.get_shape() != _source.get_shape()) {
                self.set_shape(_source.get_shape());
            }
            return _source.copy_to(self);
        },
        py::arg("source"),
        R"(
        Copy the source to this tensor. This tensor and the source should have the same element type.
        Shape will be adjusted if there is a mismatch.
    )");

    cls.def("is_continuous",
            &ov::Tensor::is_continuous,
            R"(
        Reports whether the tensor is continuous or not.
        :return: True if the tensor is continuous, otherwise False.
        :rtype: bool
    )");

    cls.def_property("shape",
                     &ov::Tensor::get_shape,
                     &ov::Tensor::set_shape,
                     R"(
                        Tensor's shape get/set.
                     )");

    cls.def_property(
        "shape",
        &ov::Tensor::get_shape,
        [](ov::Tensor& self, std::vector<size_t>& shape) {
            self.set_shape(shape);
        },
        R"(
            Tensor's shape get/set.
        )");

    cls.def("__repr__", [](const ov::Tensor& self) {
        std::stringstream ss;

        ss << "shape" << self.get_shape() << " type: " << self.get_element_type();

        return "<" + Common::get_class_name(self) + ": " + ss.str() + ">";
    });
}
