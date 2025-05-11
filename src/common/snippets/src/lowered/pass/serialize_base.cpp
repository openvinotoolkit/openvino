// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/serialize_base.hpp"

#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

SerializeBase::SerializeBase(const std::string& xml_path)
    : m_xml_path(xml_path),
      m_bin_path(get_bin_path_from_xml(xml_path)) {}

std::string SerializeBase::get_bin_path_from_xml(const std::string& xml_path) {
#if defined(__linux__)
    return "/dev/null";
#else
    return "";
#endif
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
