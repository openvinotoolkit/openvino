// ! [model_pass:model_pass_cpp]

// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/manager.hpp"

std::shared_ptr<ov::Model> model;

ov::pass::Manager manager;
manager.register_pass<ov::pass::MyModelTransformation>();
manager.run_passes(model);
// ! [model_pass:model_pass_cpp]
