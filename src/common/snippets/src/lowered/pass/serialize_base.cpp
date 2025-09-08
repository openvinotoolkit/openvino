// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/serialize_base.hpp"

#include <string>
#include <utility>

#include "snippets/itt.hpp"

namespace ov::snippets::lowered::pass {

SerializeBase::SerializeBase(std::string xml_path) : m_xml_path(std::move(xml_path)), m_bin_path(get_bin_path()) {}

std::string SerializeBase::get_bin_path() {
#if defined(__linux__)
    return "/dev/null";
#else
    return "";
#endif
}

}  // namespace ov::snippets::lowered::pass
