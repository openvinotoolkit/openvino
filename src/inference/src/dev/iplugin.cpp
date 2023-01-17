// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iplugin.hpp"

#include <cpp/ie_cnn_network.h>
#include <ie_common.h>
#include <ie_layouts.h>

#include <ie_icnn_network.hpp>
#include <ie_precision.hpp>
#include <ie_preprocess.hpp>
#include <memory>
#include <openvino/core/layout.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/preprocess/color_format.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/core/preprocess/resize_algorithm.hpp>
#include <sstream>
#include <vector>

#include "any_copy.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "converter_utils.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_icore.hpp"
#include "ie_ngraph_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "threading/ie_executor_manager.hpp"
#include "transformations/utils/utils.hpp"

ov::IPlugin::IPlugin() : m_executor_manager(InferenceEngine::executorManager()), m_is_new_api(true) {}

void ov::IPlugin::set_version(const ov::Version& version) {
    m_version = version;
}

const ov::Version& ov::IPlugin::get_version() const {
    return m_version;
}

void ov::IPlugin::set_device_name(const std::string& name) {
    m_plugin_name = name;
}

const std::string& ov::IPlugin::get_device_name() const {
    return m_plugin_name;
}

void ov::IPlugin::add_extension(const std::shared_ptr<InferenceEngine::IExtension>& extension) {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IPlugin::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any ov::IPlugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IPlugin::set_core(const std::weak_ptr<ov::ICore>& core) {
    OPENVINO_ASSERT(!core.expired());
    m_core = core;
    auto locked_core = m_core.lock();
    if (locked_core)
        m_is_new_api = locked_core->is_new_api();
}

std::shared_ptr<ov::ICore> ov::IPlugin::get_core() const {
    return m_core.lock();
}

bool ov::IPlugin::is_new_api() const {
    return m_is_new_api;
}

const std::shared_ptr<InferenceEngine::ExecutorManager>& ov::IPlugin::get_executor_manager() const {
    return m_executor_manager;
}
