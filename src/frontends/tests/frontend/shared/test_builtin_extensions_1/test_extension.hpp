// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <extension/json_transformation.hpp>
#include <openvino/frontend/node_context.hpp>

class TestExtension1 : public ov::frontend::JsonTransformationExtension {
public:
    TestExtension1();

    bool transform(const std::shared_ptr<ov::Model>& function, const std::string& config) const override;
};

ov::OutputVector CustomTranslatorCommon_1(const ov::frontend::NodeContext& node);

std::map<std::string, ov::OutputVector> CustomTranslatorCommon_2(const ov::frontend::NodeContext& node);

ov::OutputVector CustomTranslatorTensorflow(const ov::frontend::NodeContext& node);

ov::OutputVector CustomTranslatorONNX(const ov::frontend::NodeContext& node);

ov::OutputVector ReluToSwishTranslatorONNX(const ov::frontend::NodeContext& node);

std::map<std::string, ov::OutputVector> CustomTranslatorPaddle(const ov::frontend::NodeContext& node);
