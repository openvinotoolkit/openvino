// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface SerializeBase
 * @brief Base class for LinearIR serialization passes
 * @ingroup snippets
 */
class SerializeBase : public ConstPass {
public:
    OPENVINO_RTTI("SerializeBase", "", ConstPass)
    SerializeBase(const std::string& xml_path);

protected:
    std::string get_bin_path_from_xml(const std::string& xml_path);

    const std::string m_xml_path;
    const std::string m_bin_path;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
