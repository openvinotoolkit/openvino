// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/compiled_model.hpp"

#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "ie_plugin_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/icompiled_model.hpp"

#define OV_EXEC_OLD_NET_CALL_STATEMENT(...)                 \
    if (_old_impl) {                                        \
        try {                                               \
            __VA_ARGS__;                                    \
        } catch (const std::exception& ex) {                \
            throw ov::Exception(ex.what());                 \
        } catch (...) {                                     \
            OPENVINO_ASSERT(false, "Unexpected exception"); \
        }                                                   \
    }
#define OV_EXEC_NET_CALL_STATEMENT(...)                                          \
    OPENVINO_ASSERT(_impl != nullptr, "ExecutableNetwork was not initialized."); \
    try {                                                                        \
        __VA_ARGS__;                                                             \
    } catch (const std::exception& ex) {                                         \
        throw ov::Exception(ex.what());                                          \
    } catch (...) {                                                              \
        OPENVINO_ASSERT(false, "Unexpected exception");                          \
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

CompiledModel::CompiledModel(const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& impl,
                             const std::shared_ptr<void>& so)
    : _old_impl{impl},
      _so{so} {
    OPENVINO_ASSERT(_old_impl != nullptr, "CompiledModel was not initialized.");
}

std::shared_ptr<const Model> CompiledModel::get_runtime_model() const {
    OV_EXEC_OLD_NET_CALL_STATEMENT(return std::const_pointer_cast<const Model>(_old_impl->GetExecGraphInfo()));
    OV_EXEC_NET_CALL_STATEMENT(return std::const_pointer_cast<const Model>(_impl->get_runtime_model()));
}

std::vector<ov::Output<const ov::Node>> CompiledModel::inputs() const {
    OV_EXEC_OLD_NET_CALL_STATEMENT({
        std::vector<ov::Output<const ov::Node>> inputs;
        for (const auto& input : _old_impl->getInputs()) {
            inputs.emplace_back(input);
        }
        return inputs;
    });
    OV_EXEC_NET_CALL_STATEMENT({
        std::vector<ov::Output<const ov::Node>> inputs;
        for (const auto& input : _impl->inputs()) {
            inputs.emplace_back(input);
        }
        return inputs;
    });
}

ov::Output<const ov::Node> CompiledModel::input() const {
    OV_EXEC_OLD_NET_CALL_STATEMENT({
        const auto inputs = _old_impl->getInputs();
        if (inputs.size() != 1) {
            throw ov::Exception("input() must be called on a function with exactly one parameter.");
        }
        return inputs.at(0);
    });
    OV_EXEC_NET_CALL_STATEMENT({
        const auto inputs = _impl->inputs();
        if (inputs.size() != 1) {
            throw ov::Exception("input() must be called on a function with exactly one parameter.");
        }
        return inputs.at(0);
    });
}

ov::Output<const ov::Node> CompiledModel::input(size_t i) const {
    OV_EXEC_OLD_NET_CALL_STATEMENT(return _old_impl->getInputs().at(i));
    OV_EXEC_NET_CALL_STATEMENT(return _impl->inputs().at(i));
}

ov::Output<const ov::Node> CompiledModel::input(const std::string& tensor_name) const {
    OV_EXEC_OLD_NET_CALL_STATEMENT({
        for (const auto& param : _old_impl->getInputs()) {
            if (param->get_output_tensor(0).get_names().count(tensor_name)) {
                return param;
            }
        }
        throw ov::Exception("Input for tensor name '" + tensor_name + "' is not found.");
    });
    OV_EXEC_NET_CALL_STATEMENT({
        for (const auto& input : _impl->inputs()) {
            if (input.get_names().count(tensor_name)) {
                return input;
            }
        }
        throw ov::Exception("Input for tensor name '" + tensor_name + "' is not found.");
    });
}

std::vector<ov::Output<const ov::Node>> CompiledModel::outputs() const {
    OV_EXEC_OLD_NET_CALL_STATEMENT({
        std::vector<ov::Output<const ov::Node>> outputs;
        for (const auto& output : _old_impl->getOutputs()) {
            outputs.emplace_back(output);
        }
        return outputs;
    });
    OV_EXEC_NET_CALL_STATEMENT({
        std::vector<ov::Output<const ov::Node>> outputs;
        for (const auto& output : _impl->outputs()) {
            outputs.emplace_back(output);
        }
        return outputs;
    });
}
ov::Output<const ov::Node> CompiledModel::output() const {
    OV_EXEC_OLD_NET_CALL_STATEMENT({
        const auto outputs = _old_impl->getOutputs();
        if (outputs.size() != 1) {
            throw ov::Exception("output() must be called on a function with exactly one result.");
        }
        return outputs.at(0);
    });
    OV_EXEC_NET_CALL_STATEMENT({
        const auto outputs = _impl->outputs();
        if (outputs.size() != 1) {
            throw ov::Exception("output() must be called on a function with exactly one result.");
        }
        return outputs.at(0);
    });
}
ov::Output<const ov::Node> CompiledModel::output(size_t i) const {
    OV_EXEC_OLD_NET_CALL_STATEMENT(return _old_impl->getOutputs().at(i));
    OV_EXEC_NET_CALL_STATEMENT(return _impl->outputs().at(i));
}
ov::Output<const ov::Node> CompiledModel::output(const std::string& tensor_name) const {
    OV_EXEC_OLD_NET_CALL_STATEMENT({
        for (const auto& result : _old_impl->getOutputs()) {
            if (result->get_output_tensor(0).get_names().count(tensor_name)) {
                return result;
            }
        }
        throw ov::Exception("Output for tensor name '" + tensor_name + "' is not found.");
    });
    OV_EXEC_NET_CALL_STATEMENT({
        for (const auto& output : _impl->outputs()) {
            if (output.get_names().count(tensor_name)) {
                return output;
            }
        }
        throw ov::Exception("Output for tensor name '" + tensor_name + "' is not found.");
    });
}

InferRequest CompiledModel::create_infer_request() {
    OV_EXEC_OLD_NET_CALL_STATEMENT(return {_old_impl->CreateInferRequest(), _so});
    OV_EXEC_NET_CALL_STATEMENT(return {_impl->create_infer_request(), _so});
}

void CompiledModel::export_model(std::ostream& networkModel) {
    OV_EXEC_OLD_NET_CALL_STATEMENT(_old_impl->Export(networkModel));
    OV_EXEC_NET_CALL_STATEMENT(_impl->export_model(networkModel));
}

void CompiledModel::set_property(const AnyMap& config) {
    OV_EXEC_OLD_NET_CALL_STATEMENT(_old_impl->SetConfig(config));
    OV_EXEC_NET_CALL_STATEMENT(_impl->set_property(config));
}

Any CompiledModel::get_property(const std::string& name) const {
    OV_EXEC_OLD_NET_CALL_STATEMENT({
        if (ov::loaded_from_cache == name) {
            return _old_impl->isLoadedFromCache();
        }
        if (ov::supported_properties == name) {
            try {
                auto supported_properties = _old_impl->GetMetric(name).as<std::vector<PropertyName>>();
                supported_properties.erase(std::remove_if(supported_properties.begin(),
                                                          supported_properties.end(),
                                                          [](const ov::PropertyName& name) {
                                                              return name == METRIC_KEY(SUPPORTED_METRICS) ||
                                                                     name == METRIC_KEY(SUPPORTED_CONFIG_KEYS);
                                                          }),
                                           supported_properties.end());
                return supported_properties;
            } catch (ie::Exception&) {
                auto ro_properties = _old_impl->GetMetric(METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
                auto rw_properties =
                    _old_impl->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
                std::vector<ov::PropertyName> supported_properties;
                for (auto&& ro_property : ro_properties) {
                    if (ro_property != METRIC_KEY(SUPPORTED_METRICS) &&
                        ro_property != METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
                        supported_properties.emplace_back(ro_property, PropertyMutability::RO);
                    }
                }
                for (auto&& rw_property : rw_properties) {
                    supported_properties.emplace_back(rw_property, PropertyMutability::RW);
                }
                supported_properties.emplace_back(ov::supported_properties.name(), PropertyMutability::RO);
                supported_properties.emplace_back(ov::loaded_from_cache.name(), PropertyMutability::RO);
                return supported_properties;
            }
        }
        try {
            return {_old_impl->GetMetric(name), {_so}};
        } catch (ie::Exception&) {
            return {_old_impl->GetConfig(name), {_so}};
        }
    });
    OV_EXEC_NET_CALL_STATEMENT({
        return _impl->get_property(name);
        // if (ov::loaded_from_cache == name) {
        //     return _impl->isLoadedFromCache();
        // }
        // if (ov::supported_properties == name) {
        //     try {
        //         auto supported_properties = _impl->GetMetric(name).as<std::vector<PropertyName>>();
        //         supported_properties.erase(std::remove_if(supported_properties.begin(),
        //                                                   supported_properties.end(),
        //                                                   [](const ov::PropertyName& name) {
        //                                                       return name == METRIC_KEY(SUPPORTED_METRICS) ||
        //                                                              name == METRIC_KEY(SUPPORTED_CONFIG_KEYS);
        //                                                   }),
        //                                    supported_properties.end());
        //         return supported_properties;
        //     } catch (ie::Exception&) {
        //         auto ro_properties = _impl->GetMetric(METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
        //         auto rw_properties =
        //         _impl->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
        //         std::vector<ov::PropertyName> supported_properties;
        //         for (auto&& ro_property : ro_properties) {
        //             if (ro_property != METRIC_KEY(SUPPORTED_METRICS) &&
        //                 ro_property != METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        //                 supported_properties.emplace_back(ro_property, PropertyMutability::RO);
        //             }
        //         }
        //         for (auto&& rw_property : rw_properties) {
        //             supported_properties.emplace_back(rw_property, PropertyMutability::RW);
        //         }
        //         supported_properties.emplace_back(ov::supported_properties.name(), PropertyMutability::RO);
        //         supported_properties.emplace_back(ov::loaded_from_cache.name(), PropertyMutability::RO);
        //         return supported_properties;
        //     }
        // }
        // try {
        //     return {_impl->GetMetric(name), {_so}};
        // } catch (ie::Exception&) {
        //     return {_impl->GetConfig(name), {_so}};
        // }
    });
}

RemoteContext CompiledModel::get_context() const {
    OV_EXEC_OLD_NET_CALL_STATEMENT(return {_old_impl->GetContext(), {_so}});
    OV_EXEC_NET_CALL_STATEMENT(return {_impl->get_context()._impl, {_so}});
}

bool CompiledModel::operator!() const noexcept {
    return !_impl && !_old_impl;
}

CompiledModel::operator bool() const noexcept {
    return !!_impl && !!_old_impl;
}

}  // namespace ov
