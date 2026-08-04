// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include "infer_request.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace tp_gpu {

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             std::vector<ov::SoPtr<ov::ICompiledModel>>&& rank_compiled,
                             std::vector<std::string>&& device_names,
                             TPL0SharedContextPtr shared_l0_ctx,
                             TPDeviceCoordinatorPtr device_coordinator)
    : ov::ICompiledModel(model, plugin),
      m_shared_l0_ctx(std::move(shared_l0_ctx)),
      m_device_coordinator(std::move(device_coordinator)),
      m_rank_compiled(std::move(rank_compiled)),
      m_device_names(std::move(device_names)) {}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    return std::make_shared<InferRequest>(
        std::static_pointer_cast<const CompiledModel>(shared_from_this()));
}

void CompiledModel::export_model(std::ostream&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    OPENVINO_ASSERT(!m_rank_compiled.empty(),
                    "[TP_GPU] No compiled models");
    return m_rank_compiled[0]->get_runtime_model();
}

void CompiledModel::set_property(const ov::AnyMap&) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    if (name == ov::supported_properties.name()) {
        return std::vector<ov::PropertyName>{
            ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
        };
    }
    OPENVINO_THROW("[TP_GPU] Unsupported compiled model property: ", name);
}

}  // namespace tp_gpu
}  // namespace ov
