// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "intel_gpu/primitives/convert_color.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateCommonConvertColorOp(Program& p, const std::shared_ptr<ngraph::Node>& op,
                                       const cldnn::convert_color::color_format from_color,
                                       const cldnn::convert_color::color_format to_color) {
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto outDatatype = DataTypeFromPrecision(op->get_input_element_type(0));
    auto outShape = tensor_from_dims(op->get_output_shape(0));
    outShape = { outShape.sizes()[0], outShape.sizes()[2], outShape.sizes()[3], outShape.sizes()[1] };

    auto out_layout = cldnn::layout(outDatatype, cldnn::format::byxf, outShape);

    auto memory_type = cldnn::convert_color::memory_type::buffer;
    if (op->get_input_node_ptr(0)->output(0).get_rt_info().count(ov::preprocess::TensorInfoMemoryType::get_type_info_static())) {
        std::string mem_type = op->get_input_node_ptr(0)->output(0).get_rt_info().at(ov::preprocess::TensorInfoMemoryType::get_type_info_static())
                                                                                 .as<ov::preprocess::TensorInfoMemoryType>().value;
        if (mem_type.find(GPU_CONFIG_KEY(SURFACE)) != std::string::npos) {
            memory_type = cldnn::convert_color::memory_type::image;
        }
    }

    if (outShape.batch[0] > 1 && memory_type == cldnn::convert_color::memory_type::image) {
        std::vector<cldnn::primitive_id> convert_color_names;
        for (size_t b = 0; b < outShape.batch[0]; ++b) {
            cldnn::primitive::primitive_id_arr batchedInputPrimitives = { inputPrimitives[0] + "_" + std::to_string(b),
                                                                          inputPrimitives[1] + "_" + std::to_string(b)};
            cldnn::primitive_id batched_prim_id = layerName + "_" + std::to_string(b);
            convert_color_names.emplace_back(batched_prim_id);
            out_layout.size.batch[0] = 1;

            p.AddPrimitive(cldnn::convert_color(batched_prim_id,
                                                batchedInputPrimitives,
                                                from_color,
                                                to_color,
                                                memory_type,
                                                out_layout,
                                                op->get_friendly_name()));
        }
        p.AddPrimitive(cldnn::concatenation(layerName, convert_color_names, 0, op->get_friendly_name()));
    } else {
        p.AddPrimitive(cldnn::convert_color(layerName,
                                            inputPrimitives,
                                            from_color,
                                            to_color,
                                            memory_type,
                                            out_layout,
                                            op->get_friendly_name()));
    }
    p.AddPrimitiveToProfiler(op);
}

static void CreateNV12toRGBOp(Program& p, const std::shared_ptr<ngraph::op::v8::NV12toRGB>& op) {
    p.ValidateInputs(op, {1, 2});
    CreateCommonConvertColorOp(p, op, cldnn::convert_color::color_format::NV12, cldnn::convert_color::color_format::RGB);
}

static void CreateNV12toBGROp(Program& p, const std::shared_ptr<ngraph::op::v8::NV12toBGR>& op) {
    p.ValidateInputs(op, {1, 2});
    CreateCommonConvertColorOp(p, op, cldnn::convert_color::color_format::NV12, cldnn::convert_color::color_format::BGR);
}

static void CreateI420toRGBOp(Program& p, const std::shared_ptr<ngraph::op::v8::I420toRGB>& op) {
    p.ValidateInputs(op, {1, 3});
    CreateCommonConvertColorOp(p, op, cldnn::convert_color::color_format::I420, cldnn::convert_color::color_format::RGB);
}

static void CreateI420toBGROp(Program& p, const std::shared_ptr<ngraph::op::v8::I420toBGR>& op) {
    p.ValidateInputs(op, {1, 3});
    CreateCommonConvertColorOp(p, op, cldnn::convert_color::color_format::I420, cldnn::convert_color::color_format::BGR);
}

REGISTER_FACTORY_IMPL(v8, NV12toRGB);
REGISTER_FACTORY_IMPL(v8, NV12toBGR);
REGISTER_FACTORY_IMPL(v8, I420toRGB);
REGISTER_FACTORY_IMPL(v8, I420toBGR);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
