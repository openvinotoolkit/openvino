// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_transformer_app.hpp"

#include "logger/logger.hpp"

namespace transformation_sample {

ModelTransformerApp::ModelTransformerApp(const std::string& transformation_name)
    : m_transformation_name(transformation_name) {}

void ModelTransformerApp::transform(std::shared_ptr<ov::Model> model) const {
    // TODO pass pass name and implement examplary pass.
    log() << "Running App Transform" << std::endl;
    log() << "execution application defined transforamtion: " << m_transformation_name << std::endl;
}

}  // namespace transformation_sample