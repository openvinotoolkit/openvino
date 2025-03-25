// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/emitter.hpp"

#include "openvino/op/op.hpp"
#include "snippets/shape_inference/shape_inference.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface RegSpillBase
 * @brief Base class for RegSpillBegin and RegSpillEnd ops
 * @ingroup snippets
 */
class RegSpillBase : public ov::op::Op {
public:
    OPENVINO_OP("RegSpillBaseBase", "SnippetsOpset");
    RegSpillBase(const std::vector<Output<Node>>& args);
    RegSpillBase() = default;
    virtual const std::set<Reg>& get_regs_to_spill() const = 0;
    bool visit_attributes(AttributeVisitor& visitor) override;
};
class RegSpillEnd;
/**
 * @interface RegSpillBegin
 * @brief Marks the start of the register spill region.
 * @ingroup snippets
 */
class RegSpillBegin : public RegSpillBase {
public:
    OPENVINO_OP("RegSpillBegin", "SnippetsOpset", RegSpillBase);
    RegSpillBegin(std::set<Reg> regs_to_spill);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;
    std::shared_ptr<RegSpillEnd> get_reg_spill_end() const;
    const std::set<Reg>& get_regs_to_spill() const override { return m_regs_to_spill; }

    class ShapeInfer : public IShapeInferSnippets {
        size_t num_out_shapes = 0;
    public:
        explicit ShapeInfer(const std::shared_ptr<ov::Node>& n);
        Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
    };
protected:
    void validate_and_infer_types_except_RegSpillEnd();
    std::set<Reg> m_regs_to_spill = {};
};
/**
 * @interface RegSpillEnd
 * @brief Marks the end of the register spill region.
 * @ingroup snippets
 */
class RegSpillEnd : public RegSpillBase {
public:
    OPENVINO_OP("RegSpillEnd", "SnippetsOpset", RegSpillBase);
    RegSpillEnd() = default;
    RegSpillEnd(const Output<Node>& reg_spill_begin);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;
    std::shared_ptr<RegSpillBegin> get_reg_spill_begin() const {
        auto reg_spill_begin = ov::as_type_ptr<RegSpillBegin>(get_input_node_shared_ptr(0));
        OPENVINO_ASSERT(reg_spill_begin, "Can't get reg_spill_begin from reg_spill_end");
        return reg_spill_begin;
    }
    const std::set<Reg>& get_regs_to_spill() const override {
        return get_reg_spill_begin()->get_regs_to_spill();
    }
};

} // namespace op
} // namespace snippets
} // namespace ov
