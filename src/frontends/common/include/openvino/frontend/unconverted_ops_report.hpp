// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "openvino/frontend/visibility.hpp"
namespace ov {
namespace op {
namespace util {
class FrameworkNodeAttrs;
}  // namespace util
}  // namespace op
}  // namespace ov

namespace ov {
class Model;
class Node;

namespace frontend {

struct FrameworkNodeErrorInfo {
    std::string op_type;
    std::string failure_message;
};

using UnconvertedOpMap = std::map<std::string, std::string>;
using UnconvertedOpExtractor = std::function<std::optional<FrameworkNodeErrorInfo>(const std::shared_ptr<ov::Node>&)>;

FRONTEND_API UnconvertedOpMap
collect_unconverted_ops(const std::shared_ptr<ov::Model>& model,
                        const std::vector<UnconvertedOpExtractor>& custom_extractors = {});

FRONTEND_API std::string format_unconverted_ops_report(const UnconvertedOpMap& unconverted_ops,
                                                       const std::string& additional_error = std::string{},
                                                       const std::string& header = "Model wasn't fully converted.");

FRONTEND_API FrameworkNodeErrorInfo build_framework_node_error_info(const ov::op::util::FrameworkNodeAttrs& attrs);

FRONTEND_API FrameworkNodeErrorInfo build_framework_node_error_info(const ov::op::util::FrameworkNodeAttrs& attrs,
                                                                    const std::string& op_type_attr_key,
                                                                    const std::string& failure_attr_key);

FRONTEND_API FrameworkNodeErrorInfo build_framework_node_error_info(const ov::op::util::FrameworkNodeAttrs& attrs,
                                                                    const std::string& op_type_attr_key,
                                                                    const std::string& failure_attr_key,
                                                                    const std::string& version_attr_key);

template <class TNode, class Callback>
UnconvertedOpExtractor make_unconverted_op_extractor(Callback&& callback) {
    return [cb = std::forward<Callback>(callback)](
               const std::shared_ptr<ov::Node>& node) -> std::optional<FrameworkNodeErrorInfo> {
        if (const auto& typed = std::dynamic_pointer_cast<TNode>(node)) {
            return cb(typed);
        }
        return std::nullopt;
    };
}

template <class TNode>
UnconvertedOpExtractor make_framework_node_extractor(const std::string& op_type_attr_key = std::string{},
                                                     const std::string& failure_attr_key = std::string{}) {
    return make_unconverted_op_extractor<TNode>(
        [op_type_attr_key,
         failure_attr_key](const std::shared_ptr<TNode>& node) -> std::optional<FrameworkNodeErrorInfo> {
            if (!op_type_attr_key.empty() || !failure_attr_key.empty()) {
                return build_framework_node_error_info(node->get_attrs(), op_type_attr_key, failure_attr_key);
            }
            return build_framework_node_error_info(node->get_attrs());
        });
}

}  // namespace frontend
}  // namespace ov
