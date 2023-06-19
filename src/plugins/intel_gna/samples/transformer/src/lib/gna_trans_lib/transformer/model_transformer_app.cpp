// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_transformer_app.hpp"

#include <openvino/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include "logger/logger.hpp"

namespace transformation_sample {

ModelTransformerApp::ModelTransformerApp(std::shared_ptr<ov::pass::PassBase> transformation)
    : m_transformation(transformation) {}

void ModelTransformerApp::transform(std::shared_ptr<ov::Model> model) const {
    ov::pass::Manager manager;

    manager.run_passes(model);
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass_instance(m_transformation);
    manager.run_passes(model);
}

}  // namespace transformation_sample