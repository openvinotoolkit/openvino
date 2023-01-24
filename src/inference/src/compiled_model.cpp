// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/compiled_model.hpp"

#include "openvino/runtime/icompiled_model.hpp"

#define OV_COMPILED_MODEL_CALL_STATEMENT(...)                                \
    OPENVINO_ASSERT(_impl != nullptr, "CompiledModel was not initialized."); \
    try {                                                                    \
        __VA_ARGS__;                                                         \
    } catch (const std::exception& ex) {                                     \
        throw ov::Exception(ex.what());                                      \
    } catch (...) {                                                          \
        OPENVINO_ASSERT(false, "Unexpected exception");                      \
    }

namespace ov {

CompiledModel::~CompiledModel() {
    _impl = {};
}

CompiledModel::CompiledModel(const std::shared_ptr<ov::ICompiledModel>& impl, const std::shared_ptr<void>& so)
    : _impl{impl},
      _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "CompiledModel was not initialized.");
}

std::shared_ptr<const Model> CompiledModel::get_runtime_model() const {
    OV_COMPILED_MODEL_CALL_STATEMENT(return std::const_pointer_cast<const Model>(_impl->get_runtime_model()));
}

std::vector<ov::Output<const ov::Node>> CompiledModel::inputs() const {
    OV_COMPILED_MODEL_CALL_STATEMENT(return _impl->inputs(););
}

ov::Output<const ov::Node> CompiledModel::input() const {
    OV_COMPILED_MODEL_CALL_STATEMENT({
        const auto inputs = _impl->inputs();
        if (inputs.size() != 1) {
            throw ov::Exception("input() must be called on a function with exactly one parameter.");
        }
        return inputs.at(0);
    });
}

ov::Output<const ov::Node> CompiledModel::input(size_t i) const {
    OV_COMPILED_MODEL_CALL_STATEMENT(return _impl->inputs().at(i));
}

ov::Output<const ov::Node> CompiledModel::input(const std::string& tensor_name) const {
    OV_COMPILED_MODEL_CALL_STATEMENT({
        for (const auto& input : _impl->inputs()) {
            if (input.get_names().count(tensor_name)) {
                return input;
            }
        }
        throw ov::Exception("Input for tensor name '" + tensor_name + "' is not found.");
    });
}

std::vector<ov::Output<const ov::Node>> CompiledModel::outputs() const {
    OV_COMPILED_MODEL_CALL_STATEMENT(return _impl->outputs(););
}
ov::Output<const ov::Node> CompiledModel::output() const {
    OV_COMPILED_MODEL_CALL_STATEMENT({
        const auto outputs = _impl->outputs();
        if (outputs.size() != 1) {
            throw ov::Exception("output() must be called on a function with exactly one result.");
        }
        return outputs.at(0);
    });
}
ov::Output<const ov::Node> CompiledModel::output(size_t i) const {
    OV_COMPILED_MODEL_CALL_STATEMENT(return _impl->outputs().at(i));
}
ov::Output<const ov::Node> CompiledModel::output(const std::string& tensor_name) const {
    OV_COMPILED_MODEL_CALL_STATEMENT({
        for (const auto& output : _impl->outputs()) {
            if (output.get_names().count(tensor_name)) {
                return output;
            }
        }
        throw ov::Exception("Output for tensor name '" + tensor_name + "' is not found.");
    });
}

InferRequest CompiledModel::create_infer_request() {
    OV_COMPILED_MODEL_CALL_STATEMENT(return {_impl->create_infer_request(), _so});
}

void CompiledModel::export_model(std::ostream& networkModel) {
    OV_COMPILED_MODEL_CALL_STATEMENT(_impl->export_model(networkModel));
}

void CompiledModel::set_property(const AnyMap& config) {
    OV_COMPILED_MODEL_CALL_STATEMENT(_impl->set_property(config));
}

Any CompiledModel::get_property(const std::string& name) const {
    OV_COMPILED_MODEL_CALL_STATEMENT(return _impl->get_property(name););
}

RemoteContext CompiledModel::get_context() const {
    OV_COMPILED_MODEL_CALL_STATEMENT(return {_impl->get_context()._impl, {_so}});
}

bool CompiledModel::operator!() const noexcept {
    return !_impl;
}

CompiledModel::operator bool() const noexcept {
    return !!_impl;
}

}  // namespace ov

