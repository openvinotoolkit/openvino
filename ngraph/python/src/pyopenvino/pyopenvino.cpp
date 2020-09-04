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

//#include <pybind11/pybind11.h>
#include "../../pybind11/include/pybind11/pybind11.h"
#include "inference_engine/ie_blob.hpp"
#include "inference_engine/ie_core.hpp"
#include "inference_engine/ie_executable_network.hpp"
#include "inference_engine/ie_infer_request.hpp"
#include "inference_engine/ie_network.hpp"
#include "inference_engine/tensor_description.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyopenvino, m)
{
    m.doc() = "Package openvino.pyopenvino which wraps openvino C++ APIs";
    regclass_IECore(m);

    regclass_Blob<float>(m);
    // // TODO: do it the same way like Variants or somehow make trampoline to decide with Blob is called?
    // regclass_Blob<int8_t>(m);
    // regclass_Blob<uint8_t>(m);
    // regclass_Blob<int16_t>(m);
    // regclass_Blob<uint16_t>(m);
    // regclass_Blob<int32_t>(m);
    // regclass_Blob<uint32_t>(m);
    // regclass_Blob<long>(m);
    // regclass_Blob<unsigned long>(m);
    // regclass_Blob<long long>(m);
    // regclass_Blob<unsigned long long>(m);
    // regclass_Blob<double>(m);

    regclass_IENetwork(m);
    regclass_IEExecutableNetwork(m);
    regclass_InferRequest(m);
    regclass_TensorDecription(m);
}
