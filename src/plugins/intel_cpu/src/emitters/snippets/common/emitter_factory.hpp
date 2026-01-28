// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/target_machine.hpp"

namespace ov::intel_cpu {

template <typename GetHost, typename Isa, typename Wrap>
class EmitterFactory {
public:
    EmitterFactory(GetHost get_host, Isa isa, Wrap wrap)
        : get_host_(std::move(get_host)),
          isa_(isa),
          wrap_(std::move(wrap)) {}

    template <typename Emitter, typename... Args>
    [[nodiscard]] ov::snippets::jitters_value snippets(Args&&... args) const {
        auto args_tuple = std::make_tuple(std::forward<Args>(args)...);
        return {[get_host = get_host_, isa = isa_, wrap = wrap_, args_tuple = std::move(args_tuple)](
                    const ov::snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<ov::snippets::Emitter> {
                    return std::apply(
                        [&](auto&... stored) -> std::shared_ptr<ov::snippets::Emitter> {
                            auto emitter = std::make_shared<Emitter>(get_host(), isa, expr, stored...);
                            return wrap(emitter, expr);
                        },
                        args_tuple);
                },
                [](const std::shared_ptr<ov::Node>& n) -> std::set<ov::element::TypeVector> {
                    return Emitter::get_supported_precisions(n);
                }};
    }

    template <typename Emitter>
    [[nodiscard]] ov::snippets::jitters_value cpu() const {
        return {[get_host = get_host_, isa = isa_](
                    const ov::snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<ov::snippets::Emitter> {
                    return std::make_shared<Emitter>(get_host(), isa, expr->get_node());
                },
                [](const std::shared_ptr<ov::Node>& n) -> std::set<ov::element::TypeVector> {
                    return Emitter::get_supported_precisions(n);
                }};
    }

private:
    GetHost get_host_;
    Isa isa_;
    Wrap wrap_;
};

}  // namespace ov::intel_cpu
