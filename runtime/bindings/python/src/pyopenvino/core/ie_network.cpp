// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "ngraph/function.hpp"
#include <cpp/ie_cnn_network.h>
#include <ie_input_info.hpp>

#include "pyopenvino/core/ie_input_info.hpp"
#include "pyopenvino/core/ie_network.hpp"

//using PyInputsDataMap = std::map<std::string, std::shared_ptr<InferenceEngine::InputInfo>>;
//
//PYBIND11_MAKE_OPAQUE(PyInputsDataMap);

namespace py = pybind11;

void regclass_IENetwork(py::module m)
{
    py::class_<InferenceEngine::CNNNetwork, std::shared_ptr<InferenceEngine::CNNNetwork>> cls(
            m, "IENetwork");
    cls.def(py::init());

    cls.def(py::init([](std::shared_ptr<ngraph::Function>& function) {
        InferenceEngine::CNNNetwork cnnNetwork(function);
        return std::make_shared<InferenceEngine::CNNNetwork>(cnnNetwork);
    }));

    cls.def("reshape",
            [](InferenceEngine::CNNNetwork& self,
               const std::map<std::string, std::vector<size_t>>& input_shapes) {
                self.reshape(input_shapes);
            });

    cls.def("add_outputs", [](InferenceEngine::CNNNetwork& self, py::handle& outputs) {
        int i = 0;
        py::list _outputs;
        if (!py::isinstance<py::list>(outputs)) {
            if (py::isinstance<py::str>(outputs)) {
                _outputs.append(outputs.cast<py::str>());
            } else if (py::isinstance<py::tuple>(outputs)) {
                _outputs.append(outputs.cast<py::tuple>());
            }
        } else {
           _outputs = outputs.cast<py::list>();
        }
        for (py::handle output : _outputs) {
            if (py::isinstance<py::str>(_outputs[i])) {
                self.addOutput(output.cast<std::string>(), 0);
            } else if (py::isinstance<py::tuple>(output)) {
                py::tuple output_tuple = output.cast<py::tuple>();
                self.addOutput(output_tuple[0].cast<std::string>(), output_tuple[1].cast<int>());
            } else {
                IE_THROW() << "Incorrect type " <<  output.get_type() << "for layer to add at index " << i
                << ". Expected string with layer name or tuple with two elements: layer name as "
                   "first element and port id as second";
            }
            i++;
        }
    }, py::arg("outputs"));
    cls.def("add_output", &InferenceEngine::CNNNetwork::addOutput,
            py::arg("layer_name"), py::arg("output_index")=0);

    cls.def("serialize", [](InferenceEngine::CNNNetwork& self, const std::string& path_to_xml, const std::string& path_to_bin) {
        self.serialize(path_to_xml, path_to_bin);
    }, py::arg("path_to_xml"), py::arg("path_to_bin")="");

    cls.def("get_function",
            [](InferenceEngine::CNNNetwork& self) {
        return self.getFunction();
    });

    cls.def("get_ov_name_for_tensor", &InferenceEngine::CNNNetwork::getOVNameForTensor, py::arg("orig_name"));

    cls.def_property("batch_size",
                     &InferenceEngine::CNNNetwork::getBatchSize,
                     &InferenceEngine::CNNNetwork::setBatchSize);

    //auto py_inputs_data_map = py::bind_map<PyInputsDataMap>(m, "PyInputsDataMap");

//    py_inputs_data_map.def("keys", [](PyInputsDataMap& self) {
//        return py::make_key_iterator(self.begin(), self.end());
//    });

    cls.def_property_readonly("input_info", [](InferenceEngine::CNNNetwork& self) {
        std::map<std::string, std::shared_ptr<InferenceEngine::InputInfo>> inputs;
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
