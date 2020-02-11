// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <assert.h>

#include <ngraph/op/util/op_annotations.hpp>

namespace ngraph {
namespace op {
namespace util {

class EltwiseAttrs : public ngraph::op::util::OpAnnotations {
public:
    EltwiseAttrs() = default;

    explicit EltwiseAttrs(std::shared_ptr<EltwiseAttrs> & attrs):
        m_has_constant_input(attrs->has_constant_input()),
        m_consumers_count(attrs->get_consumers_count()),
        m_const_input_id(attrs->get_const_input_id()) {}

    EltwiseAttrs(size_t constant_input_id, size_t consumers_count):
        m_const_input_id(constant_input_id),
        m_has_constant_input(true),
        m_consumers_count(consumers_count) {}

    bool has_constant_input() {
        return m_has_constant_input;
    }

    size_t get_const_input_id() {
        assert(has_constant_input());
        return m_const_input_id;
    }

    void set_const_input_id(size_t val) {
        m_const_input_id = val;
        m_has_constant_input = true;
    }

    size_t get_consumers_count() {
        return m_consumers_count;
    }

    void set_consumers_count(size_t consumers_count) {
        m_consumers_count = consumers_count;
    }

    bool can_be_fused() {
        return  has_constant_input() && get_consumers_count() <= 1;
    }

    static std::shared_ptr<ngraph::op::util::EltwiseAttrs> get_op_attrs(std::shared_ptr<ngraph::op::Op> op) {
        if (!op) return nullptr;
        return std::dynamic_pointer_cast<ngraph::op::util::EltwiseAttrs>(op->get_op_annotations());
    }

    static void set_default_attrs(std::shared_ptr<ngraph::op::Op> op) {
        if (!op) return;
        op->set_op_annotations(std::make_shared<ngraph::op::util::EltwiseAttrs>());
    }

private:
    bool m_has_constant_input = false;
    size_t m_const_input_id = 0;
    size_t m_consumers_count = 0;
};

}  // namespace util
}  // namespace op
}  // namespace ngraph
