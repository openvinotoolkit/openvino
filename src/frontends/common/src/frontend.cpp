// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/util/env_util.hpp>
#include <openvino/util/file_util.hpp>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/place.hpp"
#include "openvino/util/env_util.hpp"
#include "plugin_loader.hpp"
#include "so_extension.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::frontend;

//----------- FrontEnd ---------------------------

FrontEnd::FrontEnd(const std::shared_ptr<void>& so, const std::shared_ptr<IFrontEnd>& actual)
    : m_shared_object(so),
      m_actual(actual) {}

FrontEnd::~FrontEnd() = default;

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    return m_actual->supported_impl(variants);
}

InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& params) const {
    return std::make_shared<InputModel>(m_shared_object, m_actual->load_impl(params));
}

std::shared_ptr<Model> FrontEnd::convert(const InputModel::Ptr& model) const {
    auto ov_model = m_actual->convert(model->m_actual);
    // Recreate ov::Model using main runtime, not FrontEnd's one
    auto copy = std::make_shared<Model>(ov_model->get_results(),
                                        ov_model->get_sinks(),
                                        ov_model->get_parameters(),
                                        ov_model->get_variables(),
                                        ov_model->get_friendly_name());
    copy->m_shared_object = m_shared_object;
    return copy;
}

void FrontEnd::convert(const std::shared_ptr<Model>& model) const {
    m_actual->convert(model);
}

std::shared_ptr<Model> FrontEnd::convert_partially(const InputModel::Ptr& model) const {
    auto ov_model = m_actual->convert_partially(model->m_actual);
    // Recreate ov::Model using main runtime, not FrontEnd's one
    auto copy = std::make_shared<Model>(ov_model->get_results(),
                                        ov_model->get_sinks(),
                                        ov_model->get_parameters(),
                                        ov_model->get_variables(),
                                        ov_model->get_friendly_name());
    copy->m_shared_object = m_shared_object;
    return copy;
}

std::shared_ptr<Model> FrontEnd::decode(const InputModel::Ptr& model) const {
    auto ov_model = m_actual->decode(model->m_actual);
    // Recreate ov::Model using main runtime, not FrontEnd's one
    auto copy = std::make_shared<Model>(ov_model->get_results(),
                                        ov_model->get_sinks(),
                                        ov_model->get_parameters(),
                                        ov_model->get_variables(),
                                        ov_model->get_friendly_name());
    copy->m_shared_object = m_shared_object;
    return copy;
}

void FrontEnd::normalize(const std::shared_ptr<Model>& model) const {
    m_actual->normalize(model);
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    m_actual->add_extension(extension);
}

void FrontEnd::add_extension(const std::vector<std::shared_ptr<ov::Extension>>& extensions) {
    m_actual->add_extension(extensions);
}

void FrontEnd::add_extension(const std::string& library_path) {
    m_actual->add_extension(library_path);
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
void FrontEnd::add_extension(const std::wstring& library_path) {
    m_actual->add_extension(library_path);
}
#endif

std::string FrontEnd::get_name() const {
    return m_actual->get_name();
}

//----------- IFrontEnd ---------------------------

IFrontEnd::IFrontEnd() = default;

IFrontEnd::~IFrontEnd() = default;

bool IFrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    return false;
}

IInputModel::Ptr IFrontEnd::load_impl(const std::vector<ov::Any>& params) const {
    FRONT_END_NOT_IMPLEMENTED(load_impl);
}
std::shared_ptr<Model> IFrontEnd::convert(const IInputModel::Ptr& model) const {
    FRONT_END_NOT_IMPLEMENTED(convert);
}

void IFrontEnd::convert(const std::shared_ptr<Model>&) const {
    FRONT_END_NOT_IMPLEMENTED(convert);
}

std::shared_ptr<Model> IFrontEnd::convert_partially(const IInputModel::Ptr& model) const {
    FRONT_END_NOT_IMPLEMENTED(convert_partially);
}

std::shared_ptr<Model> IFrontEnd::decode(const IInputModel::Ptr& model) const {
    FRONT_END_NOT_IMPLEMENTED(decode);
}

void IFrontEnd::normalize(const std::shared_ptr<Model>& model) const {
    FRONT_END_NOT_IMPLEMENTED(normalize);
}

void IFrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    // Left unimplemented intentionally.
    // Each frontend can support own set of extensions, so this method should be implemented on the frontend side
}

void IFrontEnd::add_extension(const std::vector<std::shared_ptr<ov::Extension>>& extensions) {
    for (const auto& ext : extensions)
        add_extension(ext);
}

void IFrontEnd::add_extension(const std::string& library_path) {
    add_extension(ov::detail::load_extensions(library_path));
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
void IFrontEnd::add_extension(const std::wstring& library_path) {
    add_extension(ov::detail::load_extensions(library_path));
}
#endif

std::string IFrontEnd::get_name() const {
    return {};
}
