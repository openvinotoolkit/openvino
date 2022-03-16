// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "proxy_plugin.hpp"

ov::proxy::Plugin::Plugin() {}
ov::proxy::Plugin::~Plugin() {}

void ov::proxy::Plugin::SetConfig(const std::map<std::string, std::string>& config) {}
InferenceEngine::QueryNetworkResult ov::proxy::Plugin::QueryNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) const {
    throw ov::Exception("Not implemented");
}
InferenceEngine::IExecutableNetworkInternal::Ptr ov::proxy::Plugin::LoadExeNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    throw ov::Exception("Not implemented");
}
void ov::proxy::Plugin::AddExtension(const std::shared_ptr<InferenceEngine::IExtension>& extension) {
    throw ov::Exception("Not implemented");
}
InferenceEngine::Parameter ov::proxy::Plugin::GetConfig(
    const std::string& name,
    const std::map<std::string, InferenceEngine::Parameter>& options) const {
    throw ov::Exception("Not implemented");
}
InferenceEngine::Parameter ov::proxy::Plugin::GetMetric(
    const std::string& name,
    const std::map<std::string, InferenceEngine::Parameter>& options) const {
    throw ov::Exception("Not implemented");
}
InferenceEngine::IExecutableNetworkInternal::Ptr ov::proxy::Plugin::ImportNetwork(
    std::istream& model,
    const std::map<std::string, std::string>& config) {
    throw ov::Exception("Not implemented");
}

void ov::proxy::create_plugin(::std::shared_ptr<::InferenceEngine::IInferencePlugin>& plugin) {
    static const InferenceEngine::Version version = {{2, 1}, CI_BUILD_NUMBER, "openvino_proxy_plugin"};
    try {
        plugin = ::std::make_shared<ov::proxy::Plugin>();
    } catch (const InferenceEngine::Exception&) {
        throw;
    } catch (const std::exception& ex) {
        IE_THROW() << ex.what();
    } catch (...) {
        IE_THROW(Unexpected);
    }
    plugin->SetVersion(version);
}
