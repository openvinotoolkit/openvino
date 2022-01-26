// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

namespace MKLDNNPlugin {
class GraphComponent {
public:
    GraphComponent() = default;
    GraphComponent(const ov::NodeVector& starts, const ov::NodeVector& ends) : starts(starts), ends(ends) {}

    void add_start(const std::shared_ptr<ov::Node>& start) { starts.push_back(start); }
    void add_end(const std::shared_ptr<ov::Node>& end) { ends.push_back(end); }

    void set_starts(const ov::NodeVector& new_starts) { starts = new_starts; }
    void set_ends(const ov::NodeVector& new_ends) { ends = new_ends; }

    const ov::NodeVector& get_starts() { return starts; }
    const ov::NodeVector& get_ends() { return ends; }

private:
    ov::NodeVector starts;
    ov::NodeVector ends;
};

bool has_graph_component(const std::shared_ptr<ov::Node>& node);
std::shared_ptr<GraphComponent> get_graph_component(const std::shared_ptr<ov::Node>& node);
void update_graph_component(const std::shared_ptr<ov::Node>& node, const std::shared_ptr<GraphComponent>& graph_component);
void set_graph_component(const std::shared_ptr<ov::Node>& node, const std::shared_ptr<GraphComponent>& graph_component);

class GraphComponentAttr : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("graph_component");
    GraphComponentAttr() = default;
    GraphComponentAttr(const std::shared_ptr<GraphComponent>& _value) : value(_value) {}

    const std::shared_ptr<GraphComponent>& get_value() { return value; }
    void set_value(const std::shared_ptr<GraphComponent>& _value) { value = _value; }

private:
    std::shared_ptr<GraphComponent> value;
};
}  // namespace MKLDNNPlugin
