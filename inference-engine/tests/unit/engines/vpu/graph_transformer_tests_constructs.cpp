// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

#include <atomic>

#include <vpu/utils/io.hpp>

using namespace vpu;

TEST_F(GraphTransformerTest, CantConnectInputOutputDatas) {
    InitCompileEnv();

    auto model = CreateModel();

    auto input = model->addInputData(
            "Input",
            DataDesc(DataType::FP16, DimsOrder::NCHW, {16, 16, 3, 1}));
    model->attrs().set<int>("numInputs", 1);

    auto output = model->addOutputData(
            "Output",
            DataDesc(DataType::FP16, DimsOrder::NCHW, {16, 16, 4, 1}));
    model->attrs().set<int>("numOutputs", 1);

    ASSERT_ANY_THROW(throw 1);
    ASSERT_ANY_THROW(VPU_THROW_UNLESS(0 == 1));

    ASSERT_ANY_THROW(
    model->connectDatas()
        .parent(input)
        .child(output)
        .mode(SharedDataMode::ROI)
        .order(SharedDataOrder::ChildWritesToParent)
        .offset(DimValues())
        .done()
    ) << "Can not short connect arbitrary input/output datas";

    ASSERT_ANY_THROW(throw 1);
}

