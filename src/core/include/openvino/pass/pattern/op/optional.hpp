// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

namespace ov {
namespace pass {
namespace pattern {
namespace op {
/// A submatch on the graph value which contains optional op defined as second input
/// The match is succeed in case of full graphs matching and except only optional op.
/// otherwise fails
class OPENVINO_API Optional : public Pattern {
public:
    OPENVINO_RTTI("patternOptional");
    /// \brief creates an Or node matching one of several sub-patterns in order. Does
    /// not add node to match list.
    /// \param patterns The patterns to try for matching
    Optional(const std::vector<DiscreteTypeInfo>& type_info,
             const Output<Node>& input_value = {}) :
             Pattern({input_value}),
             optional_type(type_info) {};

    bool match_value(pattern::Matcher* matcher,
                     const Output<Node>& pattern_value,
                     const Output<Node>& graph_value) override;

    std::vector<DiscreteTypeInfo> get_optional_types() const;

protected:
    std::vector<DiscreteTypeInfo> optional_type;
};
}  // namespace op

template <class NodeType>
std::shared_ptr<Node> optional(const Output<Node>& input = {}) {
    return std::make_shared<op::Optional>({NodeType::get_type_info_static()}, input);
}

template <class NodeType>
void collect_type_info(std::vector<DiscreteTypeInfo>& info) {
    info.push_back(NodeType::get_type_info_static());
}

template <class NodeType, class... NodeTypeArgs, typename std::enable_if<sizeof...(NodeTypeArgs) != 0, bool>::type = true>
void collect_type_info(std::vector<DiscreteTypeInfo>& info) {
    collect_type_info<NodeType>(info);
    collect_type_info<NodeTypeArgs...>(info);
}

template <class... NodeTypes>
std::shared_ptr<Node> optional(const Output<Node>& input = {}) {
    std::vector<DiscreteTypeInfo> optional_type_info_vec;
    collect_type_info<NodeTypes...>(optional_type_info_vec);
    return std::make_shared<op::Optional>(optional_type_info_vec, input);
}

}  // namespace pattern
}  // namespace pass
}  // namespace ov
