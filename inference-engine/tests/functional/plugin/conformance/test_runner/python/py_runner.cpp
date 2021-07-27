// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "shared_test_classes/read_ir/read_ir.hpp"
#include "functional_test_utils/layer_test_utils/summary.hpp"
#include "conformance.hpp"

const char *ConformanceTests::targetDevice = "";
const char *ConformanceTests::targetPluginName = "";
std::vector<std::string> ConformanceTests::IRFolderPaths = {};
std::vector<std::string> ConformanceTests::disabledTests = {};
namespace py = pybind11;

class ReadIRPyWrapper : public LayerTestsDefinitions::ReadIRBase {
public:
    ReadIRPyWrapper(const std::string &target_device) {
        ConformanceTests::targetDevice = target_device.c_str();
        targetDevice = target_device.c_str();
        external_skips = true;
    }

    void TestBody() override {};

    void RunOnFunction(pybind11::object *capsule, bool skipped = false) {
        inputs = {};
        bool throw_on_failure_state = testing::GTEST_FLAG(throw_on_failure);
        testing::GTEST_FLAG(throw_on_failure) = true;
        auto *capsule_ptr = PyCapsule_GetPointer(capsule->ptr(), "ngraph_function");
        auto *function_sp = static_cast<std::shared_ptr<ngraph::Function> *>(capsule_ptr);
        if (function_sp == nullptr)
            IE_THROW() << "Capsule doesn't contain nGraph function!";
        function = *function_sp;
        threat_as_skipped = skipped;
        Run();
        testing::GTEST_FLAG(throw_on_failure) = throw_on_failure_state;
    }

    void SaveReport() {
        LayerTestsUtils::Summary::getInstance().saveReport();
    }
};

PYBIND11_MODULE(py_conf_runner, m) {
    py::class_<ReadIRPyWrapper> runner(m, "CnfRunner");
    runner.def(py::init<const std::string &>());
    runner.def("run_on_function", &ReadIRPyWrapper::RunOnFunction,
               py::arg("function"), py::arg("is_skipped") = false);
    runner.def("save_report", &ReadIRPyWrapper::SaveReport);
}