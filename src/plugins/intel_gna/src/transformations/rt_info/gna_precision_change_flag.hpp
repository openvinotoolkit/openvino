// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace rt_info {

void add_precision_change_flag(ov::Input<Node>& node, const ov::element::Type& in, const ov::element::Type& out);

void remove_precision_change_flag(ov::Input<Node>& node);

bool is_precision_changed(const ov::Input<Node>& node);

/**
 * @ingroup ie_runtime_attr_api
 * @brief GNAPrecisionChangeFlag class represents runtime info attribute that marks that precision
 * is have to be changed before operation
 */
class GNAPrecisionChangeFlag : public RuntimeAttribute {
public:
    OPENVINO_RTTI("gna_precision_change_flag", "0");

    GNAPrecisionChangeFlag(const ov::element::Type& in, const ov::element::Type& out) : in(in), out(out) {}

    bool is_copyable() const override {
        return false;
    }

    bool is_changed() {
        return in != out;
    }
private:
    ov::element::Type in;
    ov::element::Type out;
};
} // namespace rt_info
} // namespace intel_gna
} // namespace ov
