// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_gpu/plugin/remote_context.hpp"
#include "openvino/core/model.hpp"

#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/device.hpp"
#include "transformations/convert_precision.hpp"

namespace ov::intel_gpu {

class TransformationsPipeline {
public:
    explicit TransformationsPipeline(const ExecutionConfig &conf, const std::shared_ptr<RemoteContextImpl>& context)
        : config(conf), m_context(context), device_info(context->get_engine().get_device_info()) {}
    void apply(std::shared_ptr<ov::Model> func);

private:
    static bool fuse_type_to_convert(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
    void apply_low_precision_transformations(const std::shared_ptr<ov::Model>& func,
                                             ov::element::Type infer_precision,
                                             const ov::element::TypeVector& default_precisions) const;
    void apply_activation_scaling_transformations(const std::shared_ptr<ov::Model>& func,
                                                  ov::element::Type infer_precision) const;

    const ExecutionConfig& config;
    std::shared_ptr<RemoteContextImpl> m_context;
    cldnn::device_info device_info;
};

}  // namespace ov::intel_gpu
