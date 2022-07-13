// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/topk.hpp"

#include "intel_gpu/primitives/arg_max_min.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace ov {
namespace intel_gpu {

static cldnn::arg_max_min::axis_name GetAxis(int32_t axis, size_t in_rank) {
    if (in_rank == 5) {
        if (-5 <= axis && axis <= -1)
            axis += 5;

        switch (axis) {
            case 0: return cldnn::arg_max_min::axis_name::batch;
            case 1: return cldnn::arg_max_min::axis_name::feature;
            case 2: return cldnn::arg_max_min::axis_name::z;
            case 3: return cldnn::arg_max_min::axis_name::y;
            case 4: return cldnn::arg_max_min::axis_name::x;
        }
    } else {
        if (-static_cast<int32_t>(in_rank) <= axis && axis <= -1)
            axis += in_rank;

        switch (axis) {
            case 0: return cldnn::arg_max_min::axis_name::batch;
            case 1: return cldnn::arg_max_min::axis_name::feature;
            case 2: return cldnn::arg_max_min::axis_name::y;
            case 3: return cldnn::arg_max_min::axis_name::x;
        }
    }

    return cldnn::arg_max_min::axis_name::batch;
}

static void CreateTopKOp(Program& p, const std::shared_ptr<ngraph::op::v1::TopK>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    cldnn::arg_max_min::out_type otype;
    cldnn::arg_max_min::sort_type stype;

    if (op->get_mode() == ngraph::op::v1::TopK::Mode::MAX)
        otype = cldnn::arg_max_min::out_type::max;
    else
        otype = cldnn::arg_max_min::out_type::min;

    if (op->get_sort_type() == ngraph::op::v1::TopK::SortType::SORT_VALUES)
        stype = cldnn::arg_max_min::sort_type::sort_by_values;
    else
        stype = cldnn::arg_max_min::sort_type::sort_by_indices;

    uint32_t top_k = op->get_k();
    cldnn::arg_max_min::axis_name chosen_axis = GetAxis(static_cast<int32_t>(op->get_axis()),
                                                        op->get_input_shape(0).size());

    if (op->get_output_size() == 2) {
        auto mutable_precision = op->get_output_element_type(1);
        if (mutable_precision == ngraph::element::i64) {
            mutable_precision = ngraph::element::i32;
        }

        cldnn::layout mutableLayout = cldnn::layout(DataTypeFromPrecision(mutable_precision),
                                                    DefaultFormatForDims(op->get_output_shape(1).size()),
                                                    tensor_from_dims(op->get_output_shape(1)));

        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << layer_type_name_ID(op) << ": mutable data]" << std::endl;
        }
        auto shared_memory = p.GetEngine().allocate_memory(mutableLayout);

        cldnn::primitive_id argmax_mutable_id_w = layer_type_name_ID(op) + "_md_write";
        auto argmax_mutable_prim = cldnn::mutable_data(argmax_mutable_id_w,
                                                       shared_memory,
                                                       op->get_friendly_name());
        p.primitiveIDs[argmax_mutable_id_w] = argmax_mutable_id_w;
        p.AddPrimitive(argmax_mutable_prim);
        inputPrimitives.push_back(argmax_mutable_id_w);

        std::string ArgMaxLayerName = layerName + ".0";
        auto argmaxPrim = cldnn::arg_max_min(ArgMaxLayerName,
                                             inputPrimitives,
                                             otype,
                                             top_k,
                                             chosen_axis,
                                             stype,
                                             true,
                                             op->get_friendly_name(),
                                             cldnn::padding({0, 0, 0, 0}, 0),
                                             DataTypeFromPrecision(op->get_output_element_type(0)));

        p.AddPrimitive(argmaxPrim);

        cldnn::primitive_id argmax_mutable_id_r = layerName + ".1";
        auto argmax_mutable_prim_r = cldnn::mutable_data(argmax_mutable_id_r,
                                                         { ArgMaxLayerName },
                                                         shared_memory,
                                                         op->get_friendly_name());
        p.primitiveIDs[argmax_mutable_id_r] = argmax_mutable_id_r;
        p.AddPrimitive(argmax_mutable_prim_r);
        p.InitProfileInfo(ArgMaxLayerName, layer_type_lower(op));
        p.AddPrimitiveToProfiler(ArgMaxLayerName, op);
    } else if (op->get_output_size() == 1) {
        auto argmaxPrim = cldnn::arg_max_min(layerName,
                                             inputPrimitives,
                                             otype,
                                             top_k,
                                             chosen_axis,
                                             stype,
                                             true,
                                             op->get_friendly_name(),
                                             cldnn::padding({0, 0, 0, 0}, 0),
                                             DataTypeFromPrecision(op->get_output_element_type(0)));

        p.AddPrimitive(argmaxPrim);
        p.AddPrimitiveToProfiler(op);
    } else {
        IE_THROW() << op->get_friendly_name() << " Incorrect TopK outputs number";
    }
}

REGISTER_FACTORY_IMPL(v1, TopK);

}  // namespace intel_gpu
}  // namespace ov
