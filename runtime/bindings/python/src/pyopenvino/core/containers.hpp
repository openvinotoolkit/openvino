// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <string>
#include <ie_input_info.hpp>
#include "ie_data.h"
#include "ie_blob.h"

namespace py = pybind11;

namespace Containers {
    using PyInputsDataMap = std::map<std::string, std::shared_ptr<InferenceEngine::InputInfo>>;

    using PyConstInputsDataMap =
        std::map<std::string, std::shared_ptr<const InferenceEngine::InputInfo>>;

    using PyOutputsDataMap =
        std::map<std::string, std::shared_ptr<const InferenceEngine::Data>>;

    using PyResults =
        std::map<std::string, std::shared_ptr<const InferenceEngine::Blob>>;

    void regclass_PyInputsDataMap(py::module m);
    void regclass_PyConstInputsDataMap(py::module m);
    void regclass_PyOutputsDataMap(py::module m);
    void regclass_PyResults(py::module m);
}