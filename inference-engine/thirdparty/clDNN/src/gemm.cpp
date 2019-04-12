/*
// Copyright (c) 2016 Intel Corporation
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
#include "gemm_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id gemm_type_id()
{
    static primitive_type_base<gemm> instance;
    return &instance;
}


layout gemm_inst::calc_output_layout(gemm_node const& node)
{
    assert((bool)node.get_primitive()->output_data_type == false
           && "Output data type forcing is not supported for gemm_node!");
    auto input1_layout = node.input(0).get_output_layout();
    auto input2_layout = node.input(1).get_output_layout();
    bool transpose_input1 = node.get_primitive()->transpose_input1;
    bool transpose_input2 = node.get_primitive()->transpose_input2;

    if (!transpose_input1 && !transpose_input2)
        return layout(input1_layout.data_type, format::bfyx, tensor(input1_layout.size.batch[0],  1, 
                      input2_layout.size.spatial[0], input1_layout.size.spatial[1]));
    else if (!transpose_input1 && transpose_input2)
        return layout(input1_layout.data_type, format::bfyx, tensor(input1_layout.size.batch[0], 1,
            input2_layout.size.spatial[1], input1_layout.size.spatial[1]));
    else if (transpose_input1 && !transpose_input2)
        return layout(input1_layout.data_type, format::bfyx, tensor(input1_layout.size.batch[0], 1,
            input2_layout.size.spatial[0], input1_layout.size.spatial[0]));
    else
        return layout(input1_layout.data_type, format::bfyx, tensor(input1_layout.size.batch[0], 1,
            input2_layout.size.spatial[1], input1_layout.size.spatial[0]));
    
}

std::string gemm_inst::to_string(gemm_node const& node)
{
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto alpha = desc->alpha;
    auto beta = desc->beta;
    auto transpose_input1 = desc->transpose_input1 ? " true" : "false";
    auto transpose_input2 = desc->transpose_input2 ? " true" : "false";
    std::stringstream primitive_description;

    json_composite gemm_info;
    for (size_t i = 0; i < node.inputs_count(); i++)
    {
        gemm_info.add("input_" + std::to_string(i), node.input(i).id());
    }
    gemm_info.add("alpha", alpha);
    gemm_info.add("beta", beta);
    gemm_info.add("trasnpose_input1", transpose_input1);
    gemm_info.add("transpose_input2", transpose_input2);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gemm_inst::typed_primitive_inst(network_impl& network, gemm_node const& node)
    :parent(network, node)
{
    auto input_layout = node.input(0).get_output_layout();
    auto input2_layout = node.input(1).get_output_layout();
    bool transpose_input1 = node.get_primitive()->transpose_input1;
    bool transpose_input2 = node.get_primitive()->transpose_input2;

    if (!transpose_input1 && !transpose_input2)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Input1 Columns count", input_layout.size.spatial[0], "Input2 Rows count", input2_layout.size.spatial[1], "");
        if (node.inputs_count() > 2)
        {
            auto input3_layout = node.input(2).get_output_layout();
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Input3 Columns count", input3_layout.size.spatial[0], "Input2 Columns count", input2_layout.size.spatial[0], "");
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Input3 Rows count", input3_layout.size.spatial[1], "Input1 Rows count", input_layout.size.spatial[1], "");
        }
    }

    else if (!transpose_input1 && transpose_input2)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Input1 Columns count", input_layout.size.spatial[0], "Input2 Rows count", input2_layout.size.spatial[0], "");
        if (node.inputs_count() > 2)
        {
            auto input3_layout = node.input(2).get_output_layout();
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Input13 Columns count", input3_layout.size.spatial[0], "Input2 Rows count", input2_layout.size.spatial[1], "");
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Input3 Rows count", input3_layout.size.spatial[1], "Input1 Rows count", input_layout.size.spatial[1], "");
        }
    }
    else if (transpose_input1 && !transpose_input2)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Input1 Columns count", input_layout.size.spatial[1], "Input2 Rows count", input2_layout.size.spatial[1], "");
        if (node.inputs_count() > 2)
        {
            auto input3_layout = node.input(2).get_output_layout();
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Input3 Columns count", input3_layout.size.spatial[0], "Input2 Columns count", input2_layout.size.spatial[0], "");
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Input3 Rows count", input3_layout.size.spatial[1], "Input1 Columns count", input_layout.size.spatial[0], "");
        }
    }
    else
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Input1 Columns count", input_layout.size.spatial[1], "Input2 Rows count", input2_layout.size.spatial[0], "");
        if (node.inputs_count() > 2)
        {
            auto input3_layout = node.input(2).get_output_layout();
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Input3 Columns count", input3_layout.size.spatial[0], "Input2 Rows count", input2_layout.size.spatial[1], "");
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Input3 Rows count", input3_layout.size.spatial[1], "Input1 Columns count", input_layout.size.spatial[0], "");
        }
    }

}
}
