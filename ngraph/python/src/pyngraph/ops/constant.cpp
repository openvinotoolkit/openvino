//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "ngraph/op/constant.hpp"
#include "ngraph/shape.hpp"
#include "pyngraph/ops/constant.hpp"

namespace py = pybind11;

template <typename T>
std::vector<ssize_t> _get_byte_strides(const ngraph::Shape& s)
{
    std::vector<ssize_t> byte_strides;
    std::vector<size_t> element_strides = ngraph::row_major_strides(s);
    for (auto v : element_strides)
    {
        byte_strides.push_back(static_cast<ssize_t>(v) * sizeof(T));
    }
    return byte_strides;
}

template <typename T>
py::buffer_info _get_buffer_info(const ngraph::op::Constant& c)
{
    ngraph::Shape shape = c.get_shape();
    return py::buffer_info(
        const_cast<void*>(c.get_data_ptr()),               /* Pointer to buffer */
        static_cast<ssize_t>(c.get_element_type().size()), /* Size of one scalar */
        py::format_descriptor<T>::format(), /* Python struct-style format descriptor */
        static_cast<ssize_t>(shape.size()), /* Number of dimensions */
        std::vector<ssize_t>{shape.begin(), shape.end()}, /* Buffer dimensions */
        _get_byte_strides<T>(shape)                       /* Strides (in bytes) for each index */
    );
}

template <>
py::buffer_info _get_buffer_info<ngraph::float16>(const ngraph::op::Constant& c)
{
    ngraph::Shape shape = c.get_shape();
    return py::buffer_info(
        const_cast<void*>(c.get_data_ptr()),               /* Pointer to buffer */
        static_cast<ssize_t>(c.get_element_type().size()), /* Size of one scalar */
        std::string(1, 'H'),                /* Python struct-style format descriptor */
        static_cast<ssize_t>(shape.size()), /* Number of dimensions */
        std::vector<ssize_t>{shape.begin(), shape.end()}, /* Buffer dimensions */
        _get_byte_strides<ngraph::float16>(shape)         /* Strides (in bytes) for each index */
    );
}

template <typename T>
py::array _cast_vector(const ngraph::op::Constant& self)
{
    auto vec = self.cast_vector<T>();
    return py::array(vec.size(), vec.data());
}

void regclass_pyngraph_op_Constant(py::module m)
{
    py::class_<ngraph::op::Constant, std::shared_ptr<ngraph::op::Constant>, ngraph::Node> constant(
        m, "Constant", py::buffer_protocol());
    constant.doc() = "ngraph.impl.op.Constant wraps ngraph::op::Constant";
    constant.def(
        py::init<const ngraph::element::Type&, const ngraph::Shape&, const std::vector<char>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<ngraph::float16>&>());
    constant.def(
        py::init<const ngraph::element::Type&, const ngraph::Shape&, const std::vector<float>&>());
    constant.def(
        py::init<const ngraph::element::Type&, const ngraph::Shape&, const std::vector<double>&>());
    constant.def(
        py::init<const ngraph::element::Type&, const ngraph::Shape&, const std::vector<int8_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<int16_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<int32_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<int64_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<uint8_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<uint16_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<uint32_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<uint64_t>&>());

    constant.def("get_value_strings", &ngraph::op::Constant::get_value_strings);

    constant.def("get_vector", [](const ngraph::op::Constant& self) {
        auto element_type = self.get_element_type();
        if (element_type == ngraph::element::boolean)
        {
            return _cast_vector<char>(self);
        }
        else if (element_type == ngraph::element::f16)
        {
            return _cast_vector<ngraph::float16>(self);
        }
        else if (element_type == ngraph::element::f32)
        {
            return _cast_vector<float>(self);
        }
        else if (element_type == ngraph::element::f64)
        {
            return _cast_vector<double>(self);
        }
        else if (element_type == ngraph::element::i8)
        {
            return _cast_vector<int8_t>(self);
        }
        else if (element_type == ngraph::element::i16)
        {
            return _cast_vector<int16_t>(self);
        }
        else if (element_type == ngraph::element::i32)
        {
            return _cast_vector<int32_t>(self);
        }
        else if (element_type == ngraph::element::i64)
        {
            return _cast_vector<int64_t>(self);
        }
        else if (element_type == ngraph::element::u8 || element_type == ngraph::element::u1)
        {
            return _cast_vector<uint8_t>(self);
        }
        else if (element_type == ngraph::element::u16)
        {
            return _cast_vector<uint16_t>(self);
        }
        else if (element_type == ngraph::element::u32)
        {
            return _cast_vector<uint32_t>(self);
        }
        else if (element_type == ngraph::element::u64)
        {
            return _cast_vector<uint64_t>(self);
        }
        else
        {
            throw std::runtime_error("Unsupported data type!");
        }
    });

    // Provide buffer access
    constant.def_buffer([](const ngraph::op::Constant& self) -> py::buffer_info {
        auto element_type = self.get_element_type();
        if (element_type == ngraph::element::boolean)
        {
            return _get_buffer_info<char>(self);
        }
        else if (element_type == ngraph::element::f16)
        {
            return _get_buffer_info<ngraph::float16>(self);
        }
        else if (element_type == ngraph::element::f32)
        {
            return _get_buffer_info<float>(self);
        }
        else if (element_type == ngraph::element::f64)
        {
            return _get_buffer_info<double>(self);
        }
        else if (element_type == ngraph::element::i8)
        {
            return _get_buffer_info<int8_t>(self);
        }
        else if (element_type == ngraph::element::i16)
        {
            return _get_buffer_info<int16_t>(self);
        }
        else if (element_type == ngraph::element::i32)
        {
            return _get_buffer_info<int32_t>(self);
        }
        else if (element_type == ngraph::element::i64)
        {
            return _get_buffer_info<int64_t>(self);
        }
        else if (element_type == ngraph::element::u8 || element_type == ngraph::element::u1)
        {
            return _get_buffer_info<uint8_t>(self);
        }
        else if (element_type == ngraph::element::u16)
        {
            return _get_buffer_info<uint16_t>(self);
        }
        else if (element_type == ngraph::element::u32)
        {
            return _get_buffer_info<uint32_t>(self);
        }
        else if (element_type == ngraph::element::u64)
        {
            return _get_buffer_info<uint64_t>(self);
        }
        else
        {
            throw std::runtime_error("Unsupported data type!");
        }
    });
}
