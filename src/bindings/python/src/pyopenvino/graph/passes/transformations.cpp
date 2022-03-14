// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/pass/pass.hpp>
#include <openvino/pass/serialize.hpp>
#include <pybind11/pybind11.h>

#include <memory>

#include "pyopenvino/graph/passes/transformations.hpp"

void regclass_transformations(py::module m) {
    py::class_<ov::pass::Serialize, std::shared_ptr<ov::pass::Serialize>, ov::pass::PassBase> serialize(m, "Serialize");
    serialize.doc() = "openvino.impl.Serialize transformation";
    serialize.def(py::init([](const std::string & path_to_xml,
                              const std::string & path_to_bin) {
        return std::make_shared<ov::pass::Serialize>(path_to_xml, path_to_bin);
    }));
}
