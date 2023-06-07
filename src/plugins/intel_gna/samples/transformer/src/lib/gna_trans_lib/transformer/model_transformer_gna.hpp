// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "configuration/transformer_configuration.hpp"
#include "model_transformer.hpp"

namespace transformation_sample {

// write tests
class ModelTransformerGNA : public ModelTransformer {
public:
    ModelTransformerGNA(const TransformerConfiguration& config);

    void transform(std::shared_ptr<ov::Model> model) const override;

private:
    TransformerConfiguration m_config;
};

}  // namespace transformation_sample
