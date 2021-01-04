// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/normalize_l2.hpp"
#include "ngraph/op/constant.hpp"

#include "api/normalize.hpp"
#include "api/data.hpp"

namespace CLDNNPlugin {

void CreateNormalizeL2Op(Program& p, const std::shared_ptr<ngraph::op::v0::NormalizeL2>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    // params
    auto const_axis = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    if (!const_axis)
        THROW_IE_EXCEPTION << "Unsupported axis node type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";

    auto axis = const_axis->cast_vector<size_t>();
    bool across_spatial = !(axis.size() == 1 && axis[0] == 1);
    float eps = op->get_eps();

    // WA for MO outputting %.6f
    if (eps == 0.0f) {
        eps = 1e-10f;
    }

    // We create fake scale constant and fill it with ones to keep the same behavior as current primitive
    auto scale = std::make_shared<ngraph::op::v0::Constant>(op->get_output_element_type(0), ngraph::Shape{1}, std::vector<float>{1.0});
    cldnn::layout constLayout = cldnn::layout(DataTypeFromPrecision(op->get_output_element_type(0)), cldnn::format::bfyx, cldnn::tensor{1});
    auto mem = cldnn::memory::allocate(p.GetEngine(), constLayout, 0, false);
    auto tmpPointer = mem.pointer<char>();  // implicitly maps buffer - unmap in destructor
    auto buf = tmpPointer.data();
    auto bufSize = scale->get_output_tensor(0).size();

    if (bufSize != constLayout.bytes_count())
        THROW_IE_EXCEPTION << "Invalid scales buffer in NormalizeL2 op " << op->get_friendly_name();

    std::memcpy(&buf[0], scale->get_data_ptr(), bufSize);
    auto scalesName = layerName + "_cldnn_input_scales";
    p.AddPrimitive(cldnn::data(scalesName, mem));
    p.AddInnerPrimitiveToProfiler(scalesName, layerName, op);

    auto normPrim = cldnn::normalize(layerName,
                                     inputPrimitives[0],
                                     scalesName,
                                     across_spatial,
                                     eps);

    p.AddPrimitive(normPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, NormalizeL2);

}  // namespace CLDNNPlugin
