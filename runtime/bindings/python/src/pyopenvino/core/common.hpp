// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp/ie_executable_network.hpp>
#include <cpp/ie_infer_request.hpp>
#include <ie_plugin_config.hpp>
#include <ie_blob.h>
#include <ie_parameter.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include "Python.h"
#include "ie_common.h"
#include "openvino/runtime/infer_request.hpp"

namespace py = pybind11;
using output_tensor = std::map<std::string, ov::runtime::Tensor>;

namespace Common
{
    template <typename T>
    void fill_blob(const py::handle& py_array, InferenceEngine::Blob::Ptr blob)
    {
        py::array_t<T> arr = py::cast<py::array>(py_array);
        if (arr.size() != 0) {
            // blob->allocate();
            InferenceEngine::MemoryBlob::Ptr mem_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
            std::copy(
                arr.data(0), arr.data(0) + arr.size(), mem_blob->rwmap().as<T*>());
        } else {
            py::print("Empty array!");
        }
    }

    extern const std::map<ov::element::Type, py::dtype> ov_type_to_dtype;
    extern const std::map<py::str, ov::element::Type> dtype_to_ov_type;

    ov::Strides to_numpy_strides(const ov::Strides& strides, const ov::element::Type& ov_type);

    InferenceEngine::Layout get_layout_from_string(const std::string& layout);

    const std::string& get_layout_from_enum(const InferenceEngine::Layout& layout);

    PyObject* parse_parameter(const InferenceEngine::Parameter& param);

    PyObject* parse_parameter(const InferenceEngine::Parameter& param);

    bool is_TBlob(const py::handle& blob);

    const std::shared_ptr<InferenceEngine::Blob> cast_to_blob(const py::handle& blob);

    void blob_from_numpy(const py::handle& _arr, InferenceEngine::Blob::Ptr &blob);

    void set_request_blobs(InferenceEngine::InferRequest& request, const py::dict& dictonary);

    void set_request_tensors(ov::runtime::InferRequest& request, const py::dict& dictonary);

    output_tensor get_request_tensors(ov::runtime::InferRequest& request, const py::dict& dictonary);

    uint32_t get_optimal_number_of_requests(const InferenceEngine::ExecutableNetwork& actual);
}; // namespace Common
