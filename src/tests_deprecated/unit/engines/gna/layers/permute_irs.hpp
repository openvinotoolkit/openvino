// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include "single_layer_common.hpp"

namespace GNATestIRs {
    namespace Permute {
        typedef struct { std::array<int,3> order; std::array<int,3> dim; } Permute3dimCaseParam;

        inline std::string Permute3dimModel_v6(const Permute3dimCaseParam &test_param) {
            std::string ir = R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="Permute3dim_v6" version="6">
    <layers>
        <layer id="0" name="Input_1" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>__FULL_SIZE__</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Reshape_1" precision="FP32" type="Reshape">
            <data dim="__DIM0__,__DIM1__,__DIM2__"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>__FULL_SIZE__</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>__DIM0__</dim>
                    <dim>__DIM1__</dim>
                    <dim>__DIM2__</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Permute_1" precision="FP32" type="Permute">
            <data order="__ORDER0__,__ORDER1__,__ORDER2__"/>
            <input>
                <port id="0">
                    <dim>__DIM0__</dim>
                    <dim>__DIM1__</dim>
                    <dim>__DIM2__</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>__NEW_DIM0__</dim>
                    <dim>__NEW_DIM1__</dim>
                    <dim>__NEW_DIM2__</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="Reshape_2" precision="FP32" type="Reshape">
            <data dim="1,__FULL_SIZE__"/>
            <input>
                <port id="0">
                    <dim>__NEW_DIM0__</dim>
                    <dim>__NEW_DIM1__</dim>
                    <dim>__NEW_DIM2__</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>__FULL_SIZE__</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="output_fc" precision="FP32" type="FullyConnected">
            <data out-size="__FC_DIM__"/>
            <input>
                 <port id="0">
                      <dim>1</dim>
                      <dim>__FULL_SIZE__</dim>
                 </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>__FC_DIM__</dim>
                </port>
            </output>
            <blobs>
                <weights offset="0" size="__WEIGHTS_SIZE__"/>
            </blobs>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
    </edges>
</net>
    )V0G0N";

            std::array<int, 3> new_dim;
            for (int n = 0; n < 3; n++) {
                new_dim[n] = test_param.dim[test_param.order[n]];
            }
            const int full_size = test_param.dim[0] * test_param.dim[1] * test_param.dim[2];
            const int fc_dim = 2;
            REPLACE_WITH_NUM(ir, "__ORDER0__", test_param.order[0]);
            REPLACE_WITH_NUM(ir, "__ORDER1__", test_param.order[1]);
            REPLACE_WITH_NUM(ir, "__ORDER2__", test_param.order[2]);
            REPLACE_WITH_NUM(ir, "__DIM0__", test_param.dim[0]);
            REPLACE_WITH_NUM(ir, "__DIM1__", test_param.dim[1]);
            REPLACE_WITH_NUM(ir, "__DIM2__", test_param.dim[2]);
            REPLACE_WITH_NUM(ir, "__NEW_DIM0__", new_dim[0]);
            REPLACE_WITH_NUM(ir, "__NEW_DIM1__", new_dim[1]);
            REPLACE_WITH_NUM(ir, "__NEW_DIM2__", new_dim[2]);
            REPLACE_WITH_NUM(ir, "__FULL_SIZE__", full_size);
            REPLACE_WITH_NUM(ir, "__FC_DIM__", fc_dim);
            REPLACE_WITH_NUM(ir, "__WEIGHTS_SIZE__", full_size * fc_dim * sizeof(float));

            return ir;
        }
    } }
