// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/visibility.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {

/// \brief SequenceDynamicUnitSplit represents a sequence produced by splitting `data`
/// along `axis` into runtime-many, unit-length (size 1) slices, i.e. the ONNX
/// `SplitToSequence` idiom where the number of slices is only known at runtime, but
/// each individual slice is statically known to have length 1 along `axis` (e.g. a
/// per-frame "for i in range(N_dynamic): x[i]" export idiom).
///
/// Because the slice count N is not known at graph-construction time, this sequence
/// cannot be resolved into a fixed-length SequenceMark chain the way a static split
/// can. It is only recognized by SequenceArrayLowering's narrow idiom lowering: a
/// SequenceAt reading this sequence whose position is the iteration variable of the
/// same enclosing Loop that captured it. Any other consumer (SequenceLength, a
/// differently-indexed SequenceAt, etc.) is left unconverted, so it is reported by the
/// universal unconverted-ops diagnostic rather than silently producing a wrong result.
class FRONTEND_API SequenceDynamicUnitSplit : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("SequenceDynamicUnitSplit", "util", ov::op::util::FrameworkNode);

    SequenceDynamicUnitSplit(const Output<Node>& data, int64_t axis);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    Output<Node> get_data() const {
        return input_value(0);
    }

    int64_t get_axis() const {
        return m_axis;
    }

private:
    int64_t m_axis;
};

}  // namespace frontend
}  // namespace ov
