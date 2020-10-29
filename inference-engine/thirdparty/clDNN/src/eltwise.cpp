/*
// Copyright (c) 2016-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "eltwise_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>
#include <vector>
#include <algorithm>

namespace cldnn {
primitive_type_id eltwise::type_id() {
    static primitive_type_base<eltwise> instance;
    return &instance;
}

layout eltwise_inst::calc_output_layout(eltwise_node const& node) {
    auto input_node_layout = node.input().get_non_padded_output_layout();

    auto output_type = node.get_primitive()->output_data_type ? *node.get_primitive()->output_data_type : input_node_layout.data_type;

    auto size = input_node_layout.size;
    auto format = input_node_layout.format;
    for (size_t i = 1; i < node.inputs_count(); i++) {
        auto l = node.input(i).get_non_padded_output_layout();
        size = tensor::max(size, l.size);
        if (l.format == format::b_fs_zyx_fsv16)  // use optimized 5D
            format = format::b_fs_zyx_fsv16;
        else if (l.format == format::bs_fs_zyx_bsv16_fsv16)
            format = format::bs_fs_zyx_bsv16_fsv16;
    }
    auto output_layout = layout(output_type, format, size);

    auto mode = node.get_primitive()->mode;
    // list of operations supported for integer types
    if (input_node_layout.data_type == data_types::i8 || input_node_layout.data_type == data_types::u8 ||
        input_node_layout.data_type == data_types::i32 || input_node_layout.data_type == data_types::i64) {
        std::vector<eltwise_mode> eltwise_int_modes = {eltwise_mode::sum,
                                                       eltwise_mode::sub,
                                                       eltwise_mode::prod,
                                                       eltwise_mode::div,
                                                       eltwise_mode::min,
                                                       eltwise_mode::max,
                                                       eltwise_mode::mod,
                                                       eltwise_mode::eq,
                                                       eltwise_mode::ne,
                                                       eltwise_mode::lt,
                                                       eltwise_mode::le,
                                                       eltwise_mode::gt,
                                                       eltwise_mode::ge,
                                                       eltwise_mode::logic_and,
                                                       eltwise_mode::logic_or,
                                                       eltwise_mode::logic_xor};
        if (std::find(eltwise_int_modes.begin(), eltwise_int_modes.end(), mode) == eltwise_int_modes.end())
            CLDNN_ERROR_MESSAGE(node.id(), "Requested eltwise mode is not supported for integer types.");
    }

    // Logic and comparison operations should return i8 for any inputs
    std::vector<eltwise_mode> eltwise_bool_modes = {eltwise_mode::eq,
                                                    eltwise_mode::ne,
                                                    eltwise_mode::lt,
                                                    eltwise_mode::le,
                                                    eltwise_mode::gt,
                                                    eltwise_mode::ge,
                                                    eltwise_mode::logic_and,
                                                    eltwise_mode::logic_or,
                                                    eltwise_mode::logic_xor};
    if (std::find(eltwise_bool_modes.begin(), eltwise_bool_modes.end(), mode) != eltwise_bool_modes.end()) {
        output_layout.data_type = data_types::i8;
    }

    if (node.get_primitive()->output_data_type) {
        output_layout.data_type = *node.get_primitive()->output_data_type;
    }

    if (node.has_fused_primitives()) {
        output_layout.data_type = node.get_fused_output_layout().data_type;
    }

    auto eltw = std::static_pointer_cast<const eltwise>((node.get_primitive()));
    if (!eltw->stride.empty()) {
        // we can safely use only first stride, since we're using first input, and input / stride should give exact same
        // value for every input
        input_node_layout.size.spatial[0] = (input_node_layout.size.spatial[0] - 1) / eltw->stride[0].spatial[0] + 1;
        input_node_layout.size.spatial[1] = (input_node_layout.size.spatial[1] - 1) / eltw->stride[0].spatial[1] + 1;
        input_node_layout.size.spatial[2] = (input_node_layout.size.spatial[2] - 1) / eltw->stride[0].spatial[2] + 1;
        return input_node_layout;
    }
    return output_layout;
}

static inline std::string stringify_vector(const std::vector<float>& v) {
    std::stringstream s;

    s << "{ ";

    for (size_t i = 0; i < v.size(); ++i) {
        s << v.at(i);
        if (i + 1 < v.size())
            s << ", ";
    }

    s << " }";

    return s.str();
}

std::string eltwise_inst::to_string(eltwise_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;
    std::string str_mode;

    switch (desc->mode) {
        case eltwise_mode::sum:
            str_mode = "sum";
            break;
        case eltwise_mode::sub:
            str_mode = "subtract";
            break;
        case eltwise_mode::max:
            str_mode = "max";
            break;
        case eltwise_mode::prod:
            str_mode = "product";
            break;
        case eltwise_mode::div:
            str_mode = "div";
            break;
        case eltwise_mode::min:
            str_mode = "min";
            break;
        case eltwise_mode::pow:
            str_mode = "pow";
            break;
        case eltwise_mode::squared_diff:
            str_mode = "squared_diff";
            break;
        case eltwise_mode::mod:
            str_mode = "mod";
            break;
        case eltwise_mode::eq:
            str_mode = "equal";
            break;
        case eltwise_mode::ne:
            str_mode = "not equal";
            break;
        case eltwise_mode::lt:
            str_mode = "less";
            break;
        case eltwise_mode::le:
            str_mode = "less-or-equal";
            break;
        case eltwise_mode::gt:
            str_mode = "greater";
            break;
        case eltwise_mode::ge:
            str_mode = "greater-or-equal";
            break;
        case eltwise_mode::logic_and:
            str_mode = "and";
            break;
        case eltwise_mode::logic_or:
            str_mode = "or";
            break;
        case eltwise_mode::logic_xor:
            str_mode = "xor";
            break;
        case eltwise_mode::floor_mod:
            str_mode = "floor_mod";
            break;
        default:
            str_mode = "not supported mode";
            break;
    }

    json_composite eltwise_info;
    for (size_t i = 0; i < node.inputs_count(); i++) {
        eltwise_info.add("input_" + std::to_string(i), node.input(i).id());
    }
    eltwise_info.add("mode", str_mode);
    if (desc->mode == eltwise_mode::sum) {
        eltwise_info.add("coefficients", stringify_vector(desc->coefficients));
    }
    node_info->add("eltwise info", eltwise_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

eltwise_inst::typed_primitive_inst(network_impl& network, eltwise_node const& node) : parent(network, node) {
    check_inputs_count(node);
    // check for stride
    auto prim = node.get_primitive();
    auto inputs_count = node.inputs_count();

    if (!prim->stride.empty()) {
        // number of strides must match number of inputs
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Eltwise inputs count",
                              inputs_count,
                              "Eltwise strides count",
                              prim->stride.size(),
                              "");

        const auto out_x = node.get_output_layout().size.spatial[0];
        const auto out_y = node.get_output_layout().size.spatial[1];
        // check if strides are correctly set. I.e INPUT_SIZE_X / STRIDE_X = OUTPUT_SIZE_X, same for Y dimension
        for (size_t i = 0; i < inputs_count; i++) {
            const auto& in_layout = node.input(i).get_output_layout();
            auto stride = prim->stride[i];

            const auto in_x_div_stride_x = (in_layout.size.spatial[0] - 1) / stride.spatial[0] + 1;
            if (in_x_div_stride_x != out_x && in_x_div_stride_x != 1)
                CLDNN_ERROR_NOT_EQUAL(node.id(),
                                      "Eltwise input_x / stride_x",
                                      in_x_div_stride_x,
                                      "Eltwise output_x",
                                      out_x,
                                      "");

            const auto in_y_div_stride_y = (in_layout.size.spatial[1] - 1) / stride.spatial[1] + 1;
            if (in_y_div_stride_y != out_y && in_y_div_stride_y != 1)
                CLDNN_ERROR_NOT_EQUAL(node.id(),
                                      "Eltwise inputyx / stride_y",
                                      in_y_div_stride_y,
                                      "Eltwise output_y",
                                      out_y,
                                      "");
        }
    } else {
        std::vector<int32_t> input0_size = node.input().get_output_layout().size.raw.vector();
        for (size_t i = 1; i < inputs_count; i++) {
            std::vector<int32_t> input_size = node.input(i).get_output_layout().size.raw.vector();
            for (size_t d = 0; d < input0_size.size(); d++) {
                bool sizes_equal = input0_size[d] == input_size[d];
                bool broadcast =
                    (input0_size[d] == 1 || input_size[d] == 1) && (input0_size[d] != 1 || input_size[d] != 1);
                CLDNN_ERROR_BOOL(node.id(),
                                 "Sizes equal or broadcast is possible",
                                 !(sizes_equal || broadcast),
                                 "Invalid input shapes");
            }
        }
    }

    // Check inputs calibration factors
    if (prim->inputs_calibration_factors.size() != 0) {
        auto icf_size = prim->inputs_calibration_factors.size();

        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Eltwise inputs calibration factors number",
                              icf_size,
                              "Eltwise inputs count",
                              inputs_count,
                              "");

        for (size_t i = 0; i < icf_size; ++i) {
            auto icf_size_local = node.input_calibration_factors(i).get_output_layout().size;
            auto input_size = node.input(i).get_output_layout().size;

            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Input feature number",
                                  input_size.feature[0],
                                  "Input calibration factors number",
                                  icf_size_local.count(),
                                  "");
        }
    }

    // Check inputs quantization factors
    if (!prim->input_quantization_factors.empty()) {
        auto iqf_size = prim->input_quantization_factors.size();

        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Eltwise inputs quantization factors number",
                              iqf_size,
                              "Eltwise inputs count",
                              inputs_count,
                              "");
    }
}

void eltwise_inst::check_inputs_count(eltwise_node const& node) {
    const size_t inputs_number = node.get_primitive()->input.size();
    const eltwise_mode mode = node.get_primitive()->mode;

    switch (mode) {
        case eltwise_mode::sum:
        case eltwise_mode::sub:
        case eltwise_mode::div:
        case eltwise_mode::prod:
        case eltwise_mode::max:
        case eltwise_mode::min:
        case eltwise_mode::mod:
        case eltwise_mode::logic_and:
        case eltwise_mode::logic_or:
        case eltwise_mode::logic_xor:
            if (inputs_number < 2)
                CLDNN_ERROR_MESSAGE(node.id(),
                                    "Invalid eltwise inputs number (should be equal at least to 2). Actual: " +
                                        std::to_string(inputs_number));
            break;
        case eltwise_mode::eq:
        case eltwise_mode::ne:
        case eltwise_mode::lt:
        case eltwise_mode::le:
        case eltwise_mode::gt:
        case eltwise_mode::ge:
        case eltwise_mode::squared_diff:
        case eltwise_mode::pow:
        case eltwise_mode::floor_mod:
            if (inputs_number != 2)
                CLDNN_ERROR_MESSAGE(
                    node.id(),
                    "Invalid eltwise inputs number (should be equal to 2). Actual: " + std::to_string(inputs_number));
            break;
    }
}
}  // namespace cldnn
