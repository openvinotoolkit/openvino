// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ie_blob.h"
#include "ie_common.h"
#include "ie_layouts.h"
#include "ie_precision.hpp"

#include "pyopenvino/inference_engine/tensor_description.hpp"

namespace py = pybind11;

void regclass_Blob(py::module m);

template <typename T>
void regclass_TBlob(py::module m, std::string typestring)
{
    auto pyclass_name = py::detail::c_str((std::string("TBlob") + typestring));

    py::class_<InferenceEngine::TBlob<T>, std::shared_ptr<InferenceEngine::TBlob<T>>> cls(
            m, pyclass_name);

    cls.def(py::init(
            [](const InferenceEngine::TensorDesc& tensorDesc, py::array_t<T>& arr, size_t size = 0) {
                auto blob = InferenceEngine::make_shared_blob<T>(tensorDesc);
                blob->allocate();
                if (size != 0) {
                    std::copy(arr.data(0), arr.data(0) + size, blob->rwmap().template as<T *>());
                }
                return blob;
            }));

    cls.def_property_readonly("buffer", [](InferenceEngine::TBlob<T>& self) {
        auto blob_ptr = self.buffer().template as<T*>();
        auto shape = self.getTensorDesc().getDims();
        return py::array_t<T>(shape, &blob_ptr[0], py::cast(self));
    });

    cls.def_property_readonly("tensor_desc",
                              [](InferenceEngine::TBlob<T>& self) { return self.getTensorDesc(); });
}
