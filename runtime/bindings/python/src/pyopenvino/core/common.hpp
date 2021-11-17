// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <ie_plugin_config.hpp>
#include <ie_parameter.hpp>
#include <openvino/core/type/element_type.hpp>
#include "Python.h"
#include "ie_common.h"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/executable_network.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "pyopenvino/core/containers.hpp"

namespace py = pybind11;

namespace Common
{
    const std::map<ov::element::Type, py::dtype>& ov_type_to_dtype();

    const std::map<py::str, ov::element::Type>& dtype_to_ov_type();

    ov::runtime::Tensor tensor_from_numpy(py::array& array, bool shared_memory);

    py::array as_contiguous(py::array& array, ov::element::Type type);

    const ov::runtime::Tensor& cast_to_tensor(const py::handle& tensor);

    const Containers::TensorNameMap cast_to_tensor_name_map(const py::dict& inputs);

    const Containers::TensorIndexMap cast_to_tensor_index_map(const py::dict& inputs);

    void set_request_tensors(ov::runtime::InferRequest& request, const py::dict& inputs);

    PyObject* parse_parameter(const InferenceEngine::Parameter& param);

    uint32_t get_optimal_number_of_requests(const ov::runtime::ExecutableNetwork& actual);
}; // namespace Common
