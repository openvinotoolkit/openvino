// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "model_transformer.hpp"
#include <openvino/pass/pass.hpp>

namespace transformation_sample {

// TODO write tests
class ModelTransformerApp : public ModelTransformer {
public:
    ModelTransformerApp(std::shared_ptr<ov::pass::PassBase> transformation);

    void transform(std::shared_ptr<ov::Model> model) const override;

private:
    std::shared_ptr<ov::pass::PassBase> m_transformation;
};
}  // namespace transformation_sample
