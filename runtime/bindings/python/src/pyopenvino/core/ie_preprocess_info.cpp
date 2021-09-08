// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/stl.h>

#include "pyopenvino/core/ie_preprocess_info.hpp"
#include "pyopenvino/core/common.hpp"

#include <ie_preprocess.hpp>
#include <ie_common.h>

namespace py = pybind11;

void regclass_PreProcessInfo(py::module m) {
    py::class_<InferenceEngine::PreProcessChannel, std::shared_ptr<InferenceEngine::PreProcessChannel>>(m, "PreProcessChannel")
              .def_readwrite("std_scale", &InferenceEngine::PreProcessChannel::stdScale)
              .def_readwrite("mean_value", &InferenceEngine::PreProcessChannel::meanValue)
              .def_readwrite("mean_data", &InferenceEngine::PreProcessChannel::meanData);


    py::class_<InferenceEngine::PreProcessInfo> cls(m, "PreProcessInfo");

    cls.def(py::init());
    cls.def("__getitem__", [](InferenceEngine::PreProcessInfo& self, size_t& index) {
        return self[index];
    });
    cls.def("get_number_of_channels", &InferenceEngine::PreProcessInfo::getNumberOfChannels);
    cls.def("init", &InferenceEngine::PreProcessInfo::init);
    cls.def("set_mean_image", [](InferenceEngine::PreProcessInfo& self,
                                 py::handle meanImage) {
        self.setMeanImage(Common::convert_to_blob(meanImage));
    });
    cls.def("set_mean_image_for_channel", [](InferenceEngine::PreProcessInfo& self,
                                             py::handle meanImage,
                                             const size_t channel) {
        self.setMeanImageForChannel(Common::convert_to_blob(meanImage), channel);
    });
    cls.def_property("mean_variant", &InferenceEngine::PreProcessInfo::getMeanVariant,
                     &InferenceEngine::PreProcessInfo::setVariant);
    cls.def_property("resize_algorithm", &InferenceEngine::PreProcessInfo::getResizeAlgorithm,
                     &InferenceEngine::PreProcessInfo::setResizeAlgorithm);
    cls.def_property("color_format", &InferenceEngine::PreProcessInfo::getColorFormat,
                     &InferenceEngine::PreProcessInfo::setColorFormat);

    py::enum_<InferenceEngine::MeanVariant>(m, "MeanVariant")
            .value("MEAN_IMAGE", InferenceEngine::MeanVariant::MEAN_IMAGE)
            .value("MEAN_VALUE", InferenceEngine::MeanVariant::MEAN_VALUE)
            .value("NONE", InferenceEngine::MeanVariant::NONE)
            .export_values();

    py::enum_<InferenceEngine::ResizeAlgorithm>(m, "ResizeAlgorithm")
            .value("NO_RESIZE", InferenceEngine::ResizeAlgorithm::NO_RESIZE)
            .value("RESIZE_BILINEAR", InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR)
            .value("RESIZE_AREA", InferenceEngine::ResizeAlgorithm::RESIZE_AREA)
            .export_values();

    py::enum_<InferenceEngine::ColorFormat>(m, "ColorFormat")
            .value("RAW", InferenceEngine::ColorFormat::RAW)
            .value("RGB", InferenceEngine::ColorFormat::RGB)
            .value("BGR", InferenceEngine::ColorFormat::BGR)
            .value("RGBX", InferenceEngine::ColorFormat::RGBX)
            .value("BGRX", InferenceEngine::ColorFormat::BGRX)
            .value("NV12", InferenceEngine::ColorFormat::NV12)
            .value("I420", InferenceEngine::ColorFormat::I420)
            .export_values();
}
