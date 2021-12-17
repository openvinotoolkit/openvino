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

FrontEnd::FrontEnd() = default;

FrontEnd::~FrontEnd() = default;

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    return false;
}

InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& params) const {
    FRONT_END_NOT_IMPLEMENTED(load_impl);
}
std::shared_ptr<Model> FrontEnd::convert(const InputModel::Ptr& model) const {
    FRONT_END_NOT_IMPLEMENTED(convert);
}

void FrontEnd::convert(const std::shared_ptr<Model>&) const {
    FRONT_END_NOT_IMPLEMENTED(convert);
}

std::shared_ptr<Model> FrontEnd::convert_partially(const InputModel::Ptr& model) const {
    FRONT_END_NOT_IMPLEMENTED(convert_partially);
}

std::shared_ptr<Model> FrontEnd::decode(const InputModel::Ptr& model) const {
    FRONT_END_NOT_IMPLEMENTED(decode);
}

void FrontEnd::normalize(const std::shared_ptr<Model>& model) const {
    FRONT_END_NOT_IMPLEMENTED(normalize);
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    // Left unimplemented intentionally.
    // Each frontend can support own set of extensions, so this method should be implemented on the frontend side
}

void FrontEnd::add_extension(const std::vector<std::shared_ptr<ov::Extension>>& extensions) {
    for (const auto& ext : extensions)
        add_extension(ext);
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
    return std::string();
}
