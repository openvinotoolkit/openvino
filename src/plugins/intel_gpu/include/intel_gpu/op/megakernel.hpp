// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

struct MegaKernelDim {
    int64_t input = -1;
    int64_t axis = 0;
    int64_t value = 0;

    static MegaKernelDim of_input(int64_t input, int64_t axis) { return {input, axis, 0}; }
    static MegaKernelDim constant(int64_t value) { return {-1, 0, value}; }

    bool operator==(const MegaKernelDim& o) const { return input == o.input && axis == o.axis && value == o.value; }
};

struct MegaKernelOutput {
    ov::element::Type type;
    std::vector<MegaKernelDim> shape;

    bool operator==(const MegaKernelOutput& o) const { return type == o.type && shape == o.shape; }
};

/// Everything model-specific about a MegaKernel op, set by the megakernel's transformation.
/// The plugin only interprets `outputs`; `params` is an opaque blob handed to the megakernel library as-is.
struct MegaKernelAttrs {
    std::vector<MegaKernelOutput> outputs;
    std::vector<uint8_t> params;

    bool operator==(const MegaKernelAttrs& o) const { return outputs == o.outputs && params == o.params; }
};

inline ov::PartialShape megakernel_output_shape(const MegaKernelOutput& output,
                                                const std::vector<ov::PartialShape>& input_shapes) {
    ov::PartialShape shape;
    for (const auto& d : output.shape) {
        if (d.input < 0) {
            shape.push_back(ov::Dimension(d.value));
            continue;
        }
        OPENVINO_ASSERT(static_cast<size_t>(d.input) < input_shapes.size(), "[MegaKernel] output dim refers to missing input ", d.input);
        const auto& in = input_shapes[d.input];
        if (in.rank().is_dynamic()) {
            shape.push_back(ov::Dimension::dynamic());
            continue;
        }
        OPENVINO_ASSERT(d.axis >= 0 && d.axis < in.rank().get_length(), "[MegaKernel] output dim refers to axis ", d.axis, " of rank-",
                        in.rank().get_length(), " input ", d.input);
        shape.push_back(in[d.axis]);
    }
    return shape;
}

class MegaKernel : public ov::op::Op {
public:
    OPENVINO_OP("MegaKernel", "gpu_opset");

    MegaKernel() = default;

    MegaKernel(const ov::OutputVector& inputs, const MegaKernelAttrs& attrs);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const MegaKernelAttrs& get_attrs() const { return m_attrs; }

private:
    MegaKernelAttrs m_attrs;
};

}  // namespace ov::intel_gpu::op
