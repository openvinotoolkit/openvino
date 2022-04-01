// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/util/env_util.hpp>
#include <openvino/util/file_util.hpp>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/place.hpp"
#include "openvino/op/constant.hpp"
#include "plugin_loader.hpp"
#include "so_extension.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::frontend;

std::shared_ptr<ov::Model> FrontEnd::create_copy(const std::shared_ptr<ov::Model>& ov_model,
                                                 const std::shared_ptr<void>& shared_object) {
    auto copy = std::make_shared<Model>(ov_model->get_results(),
                                        ov_model->get_sinks(),
                                        ov_model->get_parameters(),
                                        ov_model->get_variables(),
                                        ov_model->get_friendly_name());
    copy->m_shared_object = shared_object;
    copy->get_rt_info() = ov_model->get_rt_info();
    auto params = copy->get_parameters();
    ov::ParameterVector new_params;
    new_params.reserve(params.size());
    // Static library case: need to manually create all nodes with application's context, not with frontend's one.
    // ov::clone_model can't be used here because:
    //   a) It uses Node::clone_with_new_inputs which is executed in frontend's context anyway
    //   b) cloning is not working correctly for 'custom nodes' added via extensions
    // For now, create only parameters and results here, each will hold a pointer to shared object (not allowing to
    // destroy actual frontend and extensions).
    bool need_validate = false;
    for (const auto& param : params) {
        auto new_param = std::make_shared<op::v0::Parameter>(param->get_element_type(), param->get_partial_shape());
        *new_param = *param;
        new_param->output(0).set_names(param->output(0).get_names());
        new_param->output(0).get_rt_info() = param->output(0).get_rt_info();
        auto consumers = param->output(0).get_target_inputs();
        for (auto consumer : consumers) {
            if (dynamic_cast<op::v0::Result*>(consumer.get_node())) {
                // Some result points to old parameter (Param->Result case), need to trigger revalidation
                need_validate = true;
            }
            consumer.replace_source_output(new_param->get_default_output());
        }
        new_param->m_shared_object = shared_object;
        new_params.emplace_back(new_param);
    }
    copy->m_parameters = new_params;
    if (need_validate) {
        copy->validate_nodes_and_infer_types();
    }

    auto results = copy->get_results();
    ov::ResultVector new_results;
    new_results.reserve(results.size());
    for (const auto& res : results) {
        auto new_result = std::make_shared<op::v0::Result>(res->get_input_source_output(0));
        new_result->get_rt_info() = res->get_rt_info();
        new_result->output(0).get_rt_info() = res->output(0).get_rt_info();
        new_result->input(0).get_rt_info() = res->input(0).get_rt_info();
        new_result->output(0).set_names(res->output(0).get_names());
        new_result->set_friendly_name(res->get_friendly_name());
        new_result->m_shared_object = shared_object;
        new_results.emplace_back(new_result);
    }
    copy->m_results = new_results;
    return copy;
}

FrontEnd::FrontEnd() = default;

FrontEnd::~FrontEnd() = default;

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    if (m_actual) {
        return m_actual->supported_impl(variants);
    }
    return false;
}

InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, load_impl);
    auto model = std::make_shared<InputModel>();
    model->m_shared_object = m_shared_object;
    model->m_actual = m_actual->load_impl(variants);
    return model;
}

std::shared_ptr<ov::Model> FrontEnd::convert(const InputModel::Ptr& model) const {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, convert);
    return FrontEnd::create_copy(m_actual->convert(model->m_actual), m_shared_object);
}

void FrontEnd::convert(const std::shared_ptr<Model>& model) const {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, convert);
    // TODO: model shall be replaced with it's copy created in 'openvino' context, not in frontend's library context
    m_actual->convert(model);
}

std::shared_ptr<Model> FrontEnd::convert_partially(const InputModel::Ptr& model) const {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, convert_partially);
    // TODO: creation of copy in 'openvino' library context via FrontEnd::create_copy doesn't work here as nodes can be
    // 'framework' nodes which are not copyable
    return m_actual->convert_partially(model->m_actual);
}

std::shared_ptr<Model> FrontEnd::decode(const InputModel::Ptr& model) const {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, decode);
    // TODO: creation of copy in 'openvino' library context via FrontEnd::create_copy doesn't work here as nodes can be
    // 'framework' nodes which are not copyable
    return m_actual->decode(model->m_actual);
}

void FrontEnd::normalize(const std::shared_ptr<Model>& model) const {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, normalize);
    m_actual->normalize(model);
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (m_actual) {
        add_extension_to_shared_data(m_shared_object, extension);
        m_actual->add_extension(extension);
        return;
    }
    // Left unimplemented intentionally.
    // Each frontend can support own set of extensions, so this method should be implemented on the frontend side
}

void FrontEnd::add_extension(const std::vector<std::shared_ptr<ov::Extension>>& extensions) {
    for (const auto& ext : extensions) {
        add_extension(ext);
    }
}

void FrontEnd::add_extension(const std::string& library_path) {
    add_extension(ov::detail::load_extensions(library_path));
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
void FrontEnd::add_extension(const std::wstring& library_path) {
    add_extension(ov::detail::load_extensions(library_path));
}
#endif

std::string FrontEnd::get_name() const {
    if (!m_actual) {
        return {};
    }
    return m_actual->get_name();
}
