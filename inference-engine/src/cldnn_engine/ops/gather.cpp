// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/gather.hpp"

#include "api/gather.hpp"
#include "api/reorder.hpp"

namespace CLDNNPlugin {

static cldnn::gather::gather_axis GetGatherAxis(int32_t axis, cldnn::format inputFormat) {
    if (axis == 0) {
        return cldnn::gather::gather_axis::along_b;
    } else if (axis == 1) {
        return cldnn::gather::gather_axis::along_f;
    }

    if (inputFormat == cldnn::format::bfyx) {
        switch (axis) {
            case 2: return cldnn::gather::gather_axis::along_y;
            case 3: return cldnn::gather::gather_axis::along_x;
            case -1: return cldnn::gather::gather_axis::along_y;
            case -2: return cldnn::gather::gather_axis::along_f;
            case -3: return cldnn::gather::gather_axis::along_b;
            default: IE_THROW() << "Unsupported gather axis: " << axis;
        }
    } else if (inputFormat == cldnn::format::bfzyx) {
        switch (axis) {
            case 2: return cldnn::gather::gather_axis::along_z;
            case 3: return cldnn::gather::gather_axis::along_y;
            case 4: return cldnn::gather::gather_axis::along_x;
            case -1: return cldnn::gather::gather_axis::along_y;
            case -2: return cldnn::gather::gather_axis::along_z;
            case -3: return cldnn::gather::gather_axis::along_f;
            case -4: return cldnn::gather::gather_axis::along_b;
            default: IE_THROW() << "Unsupported gather axis: " << axis;
        }
    } else if (inputFormat == cldnn::format::bfwzyx) {
        switch (axis) {
            case 2: return cldnn::gather::gather_axis::along_w;
            case 3: return cldnn::gather::gather_axis::along_z;
            case 4: return cldnn::gather::gather_axis::along_y;
            case 5: return cldnn::gather::gather_axis::along_x;
            case -1: return cldnn::gather::gather_axis::along_y;
            case -2: return cldnn::gather::gather_axis::along_z;
            case -3: return cldnn::gather::gather_axis::along_w;
            case -4: return cldnn::gather::gather_axis::along_f;
            case -5: return cldnn::gather::gather_axis::along_b;
            default: IE_THROW() << "Unsupported gather axis: " << axis;
        }
    } else {
        IE_THROW() << "Unsupported gather axis: " << axis;
    }
}

void CreateGatherOp(Program& p, const std::shared_ptr<ngraph::op::v1::Gather>& op) {
    p.ValidateInputs(op, {2, 3});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    int32_t axis = static_cast<int32_t>(op->get_axis());

    std::vector<cldnn::primitive_id> reorderedInputs;
    reorderedInputs.resize(inputPrimitives.size());

    for (size_t portIndex = 0; portIndex < inputPrimitives.size(); portIndex++) {
        auto inputDataType = DataTypeFromPrecision(op->get_input_element_type(portIndex));
        if (inputDataType == cldnn::data_types::i64) {
            // clDNN primitive does not support i64 inputs,
            // so we need additional reorders to convert them to i32
            auto reorderPrimName = inputPrimitives[portIndex] + "_" + op->get_friendly_name() + Program::m_preProcessTag;
            auto targetFormat = DefaultFormatForDims(op->get_input_shape(portIndex).size());
            auto preprocessPrim = cldnn::reorder(reorderPrimName,
                                                 inputPrimitives[portIndex],
                                                 targetFormat,
                                                 cldnn::data_types::i32);
            p.AddPrimitive(preprocessPrim);
            p.AddInnerPrimitiveToProfiler(reorderPrimName, layerName, op);
            reorderedInputs[portIndex] = reorderPrimName;
        } else {
            reorderedInputs[portIndex] = inputPrimitives[portIndex];
        }
    }

    auto outLayout = DefaultFormatForDims(op->get_output_shape(0).size());
    auto gatherPrim = cldnn::gather(layerName,
                                    reorderedInputs[0],
                                    reorderedInputs[1],
                                    GetGatherAxis(axis, DefaultFormatForDims(op->get_input_shape(0).size())),
                                    outLayout,
                                    CldnnTensorFromIEDims(op->get_output_shape(0)));

    p.AddPrimitive(gatherPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, Gather);

void CreateGatherOp(Program& p, const std::shared_ptr<ngraph::op::v7::Gather>& op) {
    p.ValidateInputs(op, {2, 3, 4});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    int32_t axis = static_cast<int32_t>(op->get_axis());

    std::vector<cldnn::primitive_id> reorderedInputs;
    reorderedInputs.resize(inputPrimitives.size());

    for (size_t portIndex = 0; portIndex < inputPrimitives.size(); portIndex++) {
        auto inputDataType = DataTypeFromPrecision(op->get_input_element_type(portIndex));
        if (inputDataType == cldnn::data_types::i64) {
            // clDNN primitive does not support i64 inputs,
            // so we need additional reorders to convert them to i32
            auto reorderPrimName = inputPrimitives[portIndex] + "_" + op->get_friendly_name() + Program::m_preProcessTag;
            auto targetFormat = DefaultFormatForDims(op->get_input_shape(portIndex).size());
            auto preprocessPrim = cldnn::reorder(reorderPrimName,
                                                 inputPrimitives[portIndex],
                                                 targetFormat,
                                                 cldnn::data_types::i32);
            p.AddPrimitive(preprocessPrim);
            p.AddInnerPrimitiveToProfiler(reorderPrimName, layerName, op);
            reorderedInputs[portIndex] = reorderPrimName;
        } else {
            reorderedInputs[portIndex] = inputPrimitives[portIndex];
        }
    }

    auto outLayout = DefaultFormatForDims(op->get_output_shape(0).size());
    auto gatherPrim = cldnn::gather(layerName,
                                    reorderedInputs[0],
                                    reorderedInputs[1],
                                    GetGatherAxis(axis, DefaultFormatForDims(op->get_input_shape(0).size())),
                                    outLayout,
                                    CldnnTensorFromIEDims(op->get_output_shape(0)),
                                    op->get_batch_dims());

    p.AddPrimitive(gatherPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v7, Gather);
}  // namespace CLDNNPlugin
