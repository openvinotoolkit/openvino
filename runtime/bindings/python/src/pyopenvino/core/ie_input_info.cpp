// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_input_info.hpp>

#include <pybind11/stl.h>
#include "common.hpp"
#include "pyopenvino/core/ie_input_info.hpp"

namespace py = pybind11;

class ConstInputInfoWrapper
{
public:
    ConstInputInfoWrapper() = default;
    ~ConstInputInfoWrapper() = default;
    const InferenceEngine::InputInfo& cref() const { return value; }

protected:
    const InferenceEngine::InputInfo& ref() { return this->value; }
    const InferenceEngine::InputInfo value = InferenceEngine::InputInfo();
};

void regclass_InputInfo(py::module m)
{
    // Workaround for constant class
    py::class_<ConstInputInfoWrapper, std::shared_ptr<ConstInputInfoWrapper>> cls_const(
        m, "InputInfoCPtr");

    cls_const.def(py::init<>());

    cls_const.def_property_readonly(
        "input_data", [](const ConstInputInfoWrapper& self) { return self.cref().getInputData(); });
    cls_const.def_property_readonly("precision", [](const ConstInputInfoWrapper& self) {
        return self.cref().getPrecision().name();
    });
    cls_const.def_property_readonly("tensor_desc", [](const ConstInputInfoWrapper& self) {
        return self.cref().getTensorDesc();
    });
    cls_const.def_property_readonly(
        "name", [](const ConstInputInfoWrapper& self) { return self.cref().name(); });
    // Mutable version
    py::class_<InferenceEngine::InputInfo, std::shared_ptr<InferenceEngine::InputInfo>> cls(
        m, "InputInfoPtr");

    cls.def(py::init<>());

    cls.def_property("input_data",
                     &InferenceEngine::InputInfo::getInputData,
                     &InferenceEngine::InputInfo::setInputData);
    cls.def_property(
        "layout",
        [](InferenceEngine::InputInfo& self) {
            return Common::get_layout_from_enum(self.getLayout());
        },
        [](InferenceEngine::InputInfo& self, const std::string& layout) {
            self.setLayout(Common::get_layout_from_string(layout));
        });
    cls.def_property(
        "precision",
        [](InferenceEngine::InputInfo& self) { return self.getPrecision().name(); },
        [](InferenceEngine::InputInfo& self, const std::string& precision) {
            self.setPrecision(InferenceEngine::Precision::FromStr(precision));
        });
    cls.def_property_readonly("tensor_desc", &InferenceEngine::InputInfo::getTensorDesc);
    cls.def_property_readonly("name", &InferenceEngine::InputInfo::name);
    cls.def_property_readonly("preprocess_info", [](InferenceEngine::InputInfo& self) {
        InferenceEngine::PreProcessInfo& preprocess = self.getPreProcess();
        return preprocess;
    });
}
