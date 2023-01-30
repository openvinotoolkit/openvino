// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/runtime_attribute.hpp>

namespace ov {
namespace intel_cpu {
namespace mixed_affinity {
bool has_num_splits(const std::shared_ptr<ov::Node>& node);
size_t get_num_splits(const std::shared_ptr<ov::Node>& node);
void set_num_splits(const std::shared_ptr<ov::Node>& node, const size_t num_splits);

class NumSplits : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("num_splits");
    NumSplits() = default;
    NumSplits(const size_t value) : value(value) {}

    const size_t get_value() { return value; }
    void set_value(const size_t _value) { value = _value; }
    std::string to_string() const override { return std::to_string(value); }
    bool operator==(const NumSplits &rhs) const { return value == rhs.value; }
    bool operator!=(const NumSplits &rhs) const { return value != rhs.value; }

private:
    size_t value = 0;
};
}  // namespace mixed_affinity
}  // namespace intel_cpu
}  // namespace ov
