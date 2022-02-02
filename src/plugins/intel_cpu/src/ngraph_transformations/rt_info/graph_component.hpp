// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

namespace MKLDNNPlugin {
bool has_graph_component(const std::shared_ptr<ov::Node>& node);
size_t get_graph_component(const std::shared_ptr<ov::Node>& node);
void set_graph_component(const std::shared_ptr<ov::Node>& node, const size_t graph_component);

class GraphComponentAttr : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("graph_component");
    GraphComponentAttr() = default;
    GraphComponentAttr(const size_t _value) : value(_value) {}

    const size_t get_value() { return value; }
    void set_value(const size_t _value) { value = _value; }
    std::string to_string() const override { return std::to_string(value); }

private:
    size_t value = 0;
};
}  // namespace MKLDNNPlugin
