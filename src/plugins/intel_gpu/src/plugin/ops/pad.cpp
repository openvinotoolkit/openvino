// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "transformations/utils/utils.hpp"

#include "ngraph/op/pad.hpp"

#include "intel_gpu/primitives/border.hpp"

namespace ov {
namespace intel_gpu {

static void CreatePadOp(Program& p, const std::shared_ptr<ngraph::op::v1::Pad>& op) {
    validate_inputs_count(op, {3, 4});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    size_t rank = std::max(op->get_input_partial_shape(0).size(), static_cast<size_t>(4));

    float pad_value = 0.f;
    if (op->get_input_size() == 4) {
        auto const_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(3));
        if (!const_node) {
            IE_THROW() << "Unsupported const node type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        ngraph::op::util::get_single_value(const_node, pad_value);
    }

    auto pads_begin = op->get_pads_begin();
    auto pads_end = op->get_pads_end();

    if (pads_begin.size() < rank) {
        size_t zeros_to_add = rank - pads_begin.size();
        pads_begin.insert(pads_begin.end(), zeros_to_add, 0);
        pads_end.insert(pads_end.end(), zeros_to_add, 0);
    }

    auto tilePrim = cldnn::border(layerName,
                                  inputPrimitives[0],
                                  pads_begin,
                                  pads_end,
                                  op->get_pad_mode(),
                                  pad_value);

    p.add_primitive(*op, tilePrim);
}

REGISTER_FACTORY_IMPL(v1, Pad);

}  // namespace intel_gpu
}  // namespace ov
