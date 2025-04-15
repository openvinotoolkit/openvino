// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/proxy/infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace proxy {
class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const ov::SoPtr<ov::ICompiledModel>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const ov::SoPtr<ov::IRemoteContext>& context)
        : ov::ICompiledModel(nullptr, plugin, context),
          m_compiled_model(model) {}

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override {
        return std::make_shared<ov::proxy::InferRequest>(
            ov::SoPtr<ov::IAsyncInferRequest>{m_compiled_model->create_infer_request(), m_compiled_model._so},
            shared_from_this());
    }

    void export_model(std::ostream& model) const override {
        m_compiled_model->export_model(model);
    }

    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_compiled_model->get_runtime_model();
    }

    void set_property(const ov::AnyMap& properties) override {
        m_compiled_model->set_property(properties);
    }

    ov::Any get_property(const std::string& name) const override {
        auto property = m_compiled_model->get_property(name);
        if (!property._so)
            property._so = m_compiled_model._so;
        return property;
    }
    const std::vector<ov::Output<const ov::Node>>& inputs() const override {
        return m_compiled_model->inputs();
    }
    const std::vector<ov::Output<const ov::Node>>& outputs() const override {
        return m_compiled_model->outputs();
    }

protected:
    /**
     * @brief Method creates infer request implementation
     *
     * @return Sync infer request
     */
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

private:
    ov::SoPtr<ov::ICompiledModel> m_compiled_model;
};

}  // namespace proxy
}  // namespace ov
