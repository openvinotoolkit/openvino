//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
void regclass_TBlob(py::module m, std::string typestring) {
    auto pyclass_name = py::detail::c_str((std::string("TBlob") + typestring));

    py::class_<InferenceEngine::TBlob<T>, std::shared_ptr<InferenceEngine::TBlob<T>>> cls(
            m, pyclass_name);

    cls.def(py::init([](const InferenceEngine::TensorDesc& tensorDesc) {
        return std::make_shared<InferenceEngine::TBlob<T>>(tensorDesc);
    }));

    cls.def(py::init([](const InferenceEngine::TensorDesc& tensorDesc, py::array_t<T> arr) {
        auto size = arr.size(); // or copy from tensorDesc getDims product?
        // py::print(arr.dtype()); // validate tensorDesc with this???
        // assert arr.size() == TensorDesc.getDims().product? ???
        T* ptr = const_cast<T*>(arr.data(0)); // Note: obligatory removal of const!
        return std::make_shared<InferenceEngine::TBlob<T>>(tensorDesc, ptr, size);
    }));

    cls.def(py::init(
            [](const InferenceEngine::TensorDesc& tensorDesc, py::array_t<T>& arr, size_t size = 0) {
                if (size == 0) {
                    size = arr.size(); // or copy from tensorDesc getDims product?
                }
                auto blob = InferenceEngine::make_shared_blob<T>(tensorDesc);
                blob->allocate();
                std::copy(arr.data(0), arr.data(0) + size, blob->rwmap().template as<T*>());
                // py::print(arr.dtype()); // validate tensorDesc with this???
                // assert arr.size() == TensorDesc.getDims().product?10 ???
                return blob;
            }));

    cls.def_property_readonly("buffer", [](InferenceEngine::TBlob<T>& self) {
        auto blob_ptr = self.buffer().template as<T*>();
        auto shape = self.getTensorDesc().getDims();
        return py::array_t<T>(shape, &blob_ptr[0], py::cast(self));
    });

    cls.def_property_readonly("tensor_desc",  [](InferenceEngine::TBlob<T>& self) {
        return self.getTensorDesc();
    });
}
