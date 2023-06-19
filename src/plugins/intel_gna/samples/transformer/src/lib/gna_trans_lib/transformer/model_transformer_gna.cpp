// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_transformer_gna.hpp"

#include "backend/gna_limitations.hpp"
#include "gna_plugin_config.hpp"
#include "gna_transformations_pipeline.hpp"

namespace transformation_sample {

ModelTransformerGNA::ModelTransformerGNA(const TransformerConfiguration& config) : m_config(config) {}

void ModelTransformerGNA::transform(std::shared_ptr<ov::Model> model) const {
    ov::intel_gna::Config config;
    config.UpdateFromMap(m_config.gna_configuration);
    ov::intel_gna::limitations::Limitations::init(config.target->get_effective_execution_target());

    ov::intel_gna::TransformationsPipeline pipeline(config);
    pipeline.apply(model, m_config.transformations_names);
}
}  // namespace transformation_sample
