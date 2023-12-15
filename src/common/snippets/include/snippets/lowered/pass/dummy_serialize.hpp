// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/lowered/loop_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

class DummySerialize : public Pass {
public:
    OPENVINO_RTTI("DummySerialize", "Pass")
    DummySerialize(std::string xml_name, std::string bin_name) :
        m_xml_name(std::move(xml_name)), m_bin_name(std::move(bin_name)) {
    }
    bool run(LinearIR& linear_ir) override {
        linear_ir.serialize(m_xml_name, m_bin_name);
        return false;
    }

private:
    std::string m_xml_name, m_bin_name;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
