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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cpp/ie_cnn_network.h>
#include <ie_input_info.hpp>

#include "pyopenvino/inference_engine/ie_input_info.hpp"
#include "pyopenvino/inference_engine/ie_network.hpp"

using PyInputsDataMap = std::map<std::string, std::shared_ptr<InferenceEngine::InputInfo>>;

PYBIND11_MAKE_OPAQUE(PyInputsDataMap);

namespace py = pybind11;

void regclass_IENetwork(py::module m)
{
    py::class_<InferenceEngine::CNNNetwork, std::shared_ptr<InferenceEngine::CNNNetwork>> cls(
        m, "IENetwork");
    cls.def(py::init([](py::object* capsule) {
        // get the underlying PyObject* which is a PyCapsule pointer
        auto* pybind_capsule_ptr = capsule->ptr();

        // extract the pointer stored in the PyCapsule under the name "ngraph_function"
        auto* capsule_ptr = PyCapsule_GetPointer(pybind_capsule_ptr, "ngraph_function");

        auto* function_sp = static_cast<std::shared_ptr<ngraph::Function>*>(capsule_ptr);
        if (function_sp == nullptr)
            THROW_IE_EXCEPTION << "Cannot create CNNNetwork from capsule! Capsule doesn't contain "
                                  "nGraph function!";

        InferenceEngine::CNNNetwork cnnNetwork(*function_sp);
        return std::make_shared<InferenceEngine::CNNNetwork>(cnnNetwork);
    }));

    cls.def("reshape",
            [](InferenceEngine::CNNNetwork& self,
               const std::map<std::string, std::vector<size_t>>& input_shapes) {
                self.reshape(input_shapes);
            });

    /*    cls.def("add_outputs", [](InferenceEngine::CNNNetwork& self, py::list input) {
            self.addOutput(input_shapes);
        });*/
    /*    cls.def("serialize", );*/
    /*    cls.def("get_function", );*/
    cls.def_property("batch_size",
                     &InferenceEngine::CNNNetwork::getBatchSize,
                     &InferenceEngine::CNNNetwork::setBatchSize);

    auto py_inputs_data_map = py::bind_map<PyInputsDataMap>(m, "PyInputsDataMap");

    py_inputs_data_map.def("keys", [](PyInputsDataMap& self) {
        return py::make_key_iterator(self.begin(), self.end());
    });

    cls.def_property_readonly("input_info", [](InferenceEngine::CNNNetwork& self) {
        PyInputsDataMap inputs;
        const InferenceEngine::InputsDataMap& inputsInfo = self.getInputsInfo();
        for (auto& in : inputsInfo) {
            inputs[in.first] = in.second;
        }
        return inputs;
    });

    cls.def_property_readonly("outputs", [](InferenceEngine::CNNNetwork& self) {
        return self.getOutputsInfo();
    });

    cls.def_property_readonly("name", &InferenceEngine::CNNNetwork::getName);
}
