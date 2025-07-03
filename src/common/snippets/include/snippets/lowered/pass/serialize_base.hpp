// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface SerializeBase
 * @brief Base class for LinearIR serialization passes
 * @ingroup snippets
 */
class SerializeBase : public ConstPass {
public:
    OPENVINO_RTTI("SerializeBase", "", ConstPass)
    explicit SerializeBase(std::string xml_path);

protected:
    static std::string get_bin_path();

    const std::string m_xml_path;
    const std::string m_bin_path;
};

}  // namespace ov::snippets::lowered::pass
