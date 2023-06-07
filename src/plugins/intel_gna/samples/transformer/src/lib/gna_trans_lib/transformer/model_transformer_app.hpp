// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "model_transformer.hpp"

namespace transformation_sample {

// print and validate available passes.
// TODO write tests
class ModelTransformerApp : public ModelTransformer {
public:
    ModelTransformerApp(const std::string& transformation_name);

    void transform(std::shared_ptr<ov::Model> model) const override;

private:
    std::string m_transformation_name;
};
}  // namespace transformation_sample
