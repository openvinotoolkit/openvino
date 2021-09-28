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
#include "inference_engine/ie_version.hpp"
#include "inference_engine/ie_parameter.hpp"
#include "inference_engine/ie_input_info.hpp"
#include "inference_engine/ie_data.hpp"

#include <string>
#include <ie_common.h>
#include <ie_version.hpp>


namespace py = pybind11;

std::string get_version() {
    auto version = InferenceEngine::GetInferenceEngineVersion();
    std::string version_str = std::to_string(version->apiVersion.major) + ".";
    version_str += std::to_string(version->apiVersion.minor) + ".";
    version_str += version->buildNumber;
    return version_str;
}

PYBIND11_MODULE(pyopenvino, m) {

    m.doc() = "Package openvino.pyopenvino which wraps openvino C++ APIs";
    m.def("get_version", &get_version);
    py::enum_<InferenceEngine::StatusCode>(m, "StatusCode")
    .value("OK", InferenceEngine::StatusCode::OK)
    .value("GENERAL_ERROR", InferenceEngine::StatusCode::GENERAL_ERROR)
    .value("NOT_IMPLEMENTED", InferenceEngine::StatusCode::NOT_IMPLEMENTED)
    .value("NETWORK_NOT_LOADED", InferenceEngine::StatusCode::NETWORK_NOT_LOADED)
    .value("PARAMETER_MISMATCH", InferenceEngine::StatusCode::PARAMETER_MISMATCH)
    .value("NOT_FOUND", InferenceEngine::StatusCode::NOT_FOUND)
    .value("OUT_OF_BOUNDS", InferenceEngine::StatusCode::OUT_OF_BOUNDS)
    .value("UNEXPECTED", InferenceEngine::StatusCode::UNEXPECTED)
    .value("REQUEST_BUSY", InferenceEngine::StatusCode::REQUEST_BUSY)
    .value("RESULT_NOT_READY", InferenceEngine::StatusCode::RESULT_NOT_READY)
    .value("NOT_ALLOCATED", InferenceEngine::StatusCode::NOT_ALLOCATED)
    .value("INFER_NOT_STARTED", InferenceEngine::StatusCode::INFER_NOT_STARTED)
    .value("NETWORK_NOT_READ", InferenceEngine::StatusCode::NETWORK_NOT_READ)
    .export_values();


    regclass_IECore(m);

    // GeneralBlob
    regclass_Blob(m);
    // Specific type Blobs
    regclass_TBlob<float>(m, "Float32");
    regclass_TBlob<int8_t>(m, "Int8");
    regclass_TBlob<uint8_t>(m, "Uint8");
    regclass_TBlob<int16_t>(m, "Int16");
    // regclass_TBlob<uint16_t>(m, "Uint16");
    regclass_TBlob<int32_t>(m, "Int32");
    // regclass_TBlob<uint32_t>(m, "Uint32");
    regclass_TBlob<long>(m, "Int64");
    // regclass_TBlob<unsigned long>(m, "UInt64");
    // regclass_TBlob<long long>(m);
    // regclass_TBlob<unsigned long long>(m);
    // regclass_TBlob<double>(m);

    regclass_IENetwork(m);
    regclass_ExecutableNetwork(m);
    regclass_InferRequest(m);
    regclass_TensorDecription(m);
    regclass_Version(m);
    regclass_Parameter(m);
    regclass_Data(m);
    regclass_InputInfo(m);
}
