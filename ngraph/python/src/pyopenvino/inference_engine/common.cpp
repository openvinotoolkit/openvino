// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <unordered_map>

#include "common.hpp"

namespace Common {

    namespace {
        const std::unordered_map<int, std::string> layout_int_to_str_map = {{0,   "ANY"},
                                                                            {1,   "NCHW"},
                                                                            {2,   "NHWC"},
                                                                            {3,   "NCDHW"},
                                                                            {4,   "NDHWC"},
                                                                            {64,  "OIHW"},
                                                                            {95,  "SCALAR"},
                                                                            {96,  "C"},
                                                                            {128, "CHW"},
                                                                            {192, "HW"},
                                                                            {193, "NC"},
                                                                            {194, "CN"},
                                                                            {200, "BLOCKED"}};

        const std::unordered_map <std::string, InferenceEngine::Layout> layout_str_to_enum = {
                {"ANY",     InferenceEngine::Layout::ANY},
                {"NHWC",    InferenceEngine::Layout::NHWC},
                {"NCHW",    InferenceEngine::Layout::NCHW},
                {"NCDHW",   InferenceEngine::Layout::NCDHW},
                {"NDHWC",   InferenceEngine::Layout::NDHWC},
                {"OIHW",    InferenceEngine::Layout::OIHW},
                {"GOIHW",   InferenceEngine::Layout::GOIHW},
                {"OIDHW",   InferenceEngine::Layout::OIDHW},
                {"GOIDHW",  InferenceEngine::Layout::GOIDHW},
                {"SCALAR",  InferenceEngine::Layout::SCALAR},
                {"C",       InferenceEngine::Layout::C},
                {"CHW",     InferenceEngine::Layout::CHW},
                {"HW",      InferenceEngine::Layout::HW},
                {"NC",      InferenceEngine::Layout::NC},
                {"CN",      InferenceEngine::Layout::CN},
                {"BLOCKED", InferenceEngine::Layout::BLOCKED}
        };
    }

    InferenceEngine::Layout get_layout_from_string(const std::string &layout) {
        return layout_str_to_enum.at(layout);
    }

    const std::string &get_layout_from_enum(const InferenceEngine::Layout &layout) {
        return layout_int_to_str_map.at(layout);
    }

    PyObject *parse_parameter(const InferenceEngine::Parameter &param) {
        // Check for std::string
        if (param.is<std::string>()) {
            return PyUnicode_FromString(param.as<std::string>().c_str());
        }
            // Check for int
        else if (param.is<int>()) {
            auto val = param.as<int>();
            return PyLong_FromLong((long) val);
        }
            // Check for unsinged int
        else if (param.is<unsigned int>()) {
            auto val = param.as<unsigned int>();
            return PyLong_FromLong((unsigned long) val);
        }
            // Check for float
        else if (param.is<float>()) {
            auto val = param.as<float>();
            return PyFloat_FromDouble((double) val);
        }
            // Check for bool
        else if (param.is<bool>()) {
            auto val = param.as<bool>();
            return val ? Py_True : Py_False;
        }
            // Check for std::vector<std::string>
        else if (param.is<std::vector<std::string>>()) {
            auto val = param.as<std::vector<std::string>>();
            PyObject *list = PyList_New(0);
            for (const auto &it : val) {
                PyObject *str_val = PyUnicode_FromString(it.c_str());
                PyList_Append(list, str_val);
            }
            return list;
        }
            // Check for std::vector<int>
        else if (param.is<std::vector<int>>()) {
            auto val = param.as<std::vector<int>>();
            PyObject *list = PyList_New(0);
            for (const auto &it : val) {
                PyList_Append(list, PyLong_FromLong(it));
            }
            return list;
        }
            // Check for std::vector<unsigned int>
        else if (param.is < std::vector < unsigned int >> ()) {
            auto val = param.as < std::vector < unsigned int >> ();
            PyObject *list = PyList_New(0);
            for (const auto &it : val) {
                PyList_Append(list, PyLong_FromLong(it));
            }
            return list;
        }
            // Check for std::vector<float>
        else if (param.is<std::vector<float>>()) {
            auto val = param.as<std::vector<float>>();
            PyObject *list = PyList_New(0);
            for (const auto &it : val) {
                PyList_Append(list, PyFloat_FromDouble((double) it));
            }
            return list;
        }
            // Check for std::tuple<unsigned int, unsigned int>
        else if (param.is<std::tuple<unsigned int, unsigned int>>()) {
            auto val = param.as<std::tuple<unsigned int, unsigned int>>();
            PyObject *tuple = PyTuple_New(2);
            PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long) std::get<0>(val)));
            PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long) std::get<1>(val)));
            return tuple;
        }
            // Check for std::tuple<unsigned int, unsigned int, unsigned int>
        else if (param.is<std::tuple<unsigned int, unsigned int, unsigned int>>()) {
            auto val = param.as<std::tuple<unsigned int, unsigned int, unsigned int>>();
            PyObject *tuple = PyTuple_New(3);
            PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long) std::get<0>(val)));
            PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long) std::get<1>(val)));
            PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLong((unsigned long) std::get<2>(val)));
            return tuple;
        }
            // Check for std::map<std::string, std::string>
        else if (param.is <std::map<std::string, std::string>>()) {
            auto val = param.as <std::map<std::string, std::string>>();
            PyObject *dict = PyDict_New();
            for (const auto &it : val) {
                PyDict_SetItemString(dict, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
            }
            return dict;
        }
            // Check for std::map<std::string, int>
        else if (param.is<std::map<std::string, int>>()) {
            auto val = param.as<std::map<std::string, int>>();
            PyObject *dict = PyDict_New();
            for (const auto &it : val) {
                PyDict_SetItemString(dict, it.first.c_str(), PyLong_FromLong((long) it.second));
            }
            return dict;
        } else {
            PyErr_SetString(PyExc_TypeError, "Failed to convert parameter to Python representation!");
            return (PyObject *) NULL;
        }
    }

    const std::shared_ptr<InferenceEngine::Blob> convert_to_blob(const py::handle& blob) {
        if (py::isinstance<InferenceEngine::TBlob<float>>(blob)) {
            return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<float>> &>();
        } else if (py::isinstance<InferenceEngine::TBlob<double>>(blob)) {
            return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<double>> &>();
        } else if (py::isinstance<InferenceEngine::TBlob<int8_t>>(blob)) {
            return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<int8_t>> &>();
        } else if (py::isinstance<InferenceEngine::TBlob<int16_t>>(blob)) {
            return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<int16_t>> &>();
        } else if (py::isinstance<InferenceEngine::TBlob<int32_t>>(blob)) {
            return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<int32_t>> &>();
        } else if (py::isinstance<InferenceEngine::TBlob<int64_t>>(blob)) {
            return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<int64_t>> &>();
        } else if (py::isinstance<InferenceEngine::TBlob<uint8_t>>(blob)) {
            return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<uint8_t>> &>();
        } else if (py::isinstance<InferenceEngine::TBlob<uint16_t>>(blob)) {
            return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<uint16_t>> &>();
        } else if (py::isinstance<InferenceEngine::TBlob<uint32_t>>(blob)) {
            return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<uint32_t>> &>();
        } else if (py::isinstance<InferenceEngine::TBlob<uint64_t>>(blob)) {
            return blob.cast<const std::shared_ptr<InferenceEngine::TBlob<uint64_t>> &>();
        } else {
            // Throw error
        }
    }

};
