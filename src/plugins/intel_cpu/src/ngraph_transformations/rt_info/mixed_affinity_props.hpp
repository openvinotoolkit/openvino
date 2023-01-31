// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include "../mixed_affinity_utils.hpp"

namespace ov {
namespace intel_cpu {
namespace mixed_affinity {

bool has_properties(const std::shared_ptr<ov::Node>& node);
Properties get_properties(const std::shared_ptr<ov::Node>& node);
void set_properties(const std::shared_ptr<ov::Node>& node, const Properties value);

class MixedAffinityProps : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("mixed_affinity_props");
    MixedAffinityProps() = default;
    MixedAffinityProps(const Properties value) : value(value) {}

    const Properties get_value() { return value; }
    void set_value(const Properties _value) { value = _value; }
    bool operator==(const MixedAffinityProps &rhs) const;
    bool operator!=(const MixedAffinityProps &rhs) const;
    std::string to_string() const override;

private:
    Properties value;
};
}  // namespace mixed_affinity
}  // namespace intel_cpu
}  // namespace ov
