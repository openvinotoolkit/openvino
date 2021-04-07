// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_input_info.hpp>

#include <pybind11/stl.h>
#include "common.hpp"
#include "pyopenvino/inference_engine/ie_input_info.hpp"

namespace py = pybind11;

template <typename T>
class ConstWrapper {
public:
    ConstWrapper() = default;
    ~ConstWrapper() = default;
    const T& cref() const { return value; }

protected:
    T& ref() { return this->value; }
    T value;
};

void regclass_InputInfo(py::module m) {
    // Workaround for constant class
    py::class_<ConstWrapper<const InferenceEngine::InputInfo>,
               std::shared_ptr<ConstWrapper<const InferenceEngine::InputInfo>>>
        cls_const(m, "InputInfoCPtr");

    cls_const.def(py::init<>());

    cls_const.def_property_readonly("input_data",
                                    [](const ConstWrapper<const InferenceEngine::InputInfo>& self) {
                                        return self.cref().getInputData();
                                    });
    cls_const.def_property_readonly("precision",
                                    [](const ConstWrapper<const InferenceEngine::InputInfo>& self) {
                                        return self.cref().getPrecision().name();
                                    });
    cls_const.def_property_readonly("tensor_desc",
                                    [](const ConstWrapper<const InferenceEngine::InputInfo>& self) {
                                        return self.cref().getTensorDesc();
                                    });
    cls_const.def_property_readonly("name",
                                    [](const ConstWrapper<const InferenceEngine::InputInfo>& self) {
                                        return self.cref().name();
                                    });
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
    cls.def_property_readonly("preprocess_info", &InferenceEngine::InputInfo::getPreProcess);
}
