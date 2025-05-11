// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "openvino/frontend/manager.hpp"

struct BaseFEParam {
    BaseFEParam() {}

    BaseFEParam(const std::string& name, const std::string& path) : m_frontEndName(name), m_modelsPath(path) {}

    std::string m_frontEndName;
    std::string m_modelsPath;
};

struct PartShape {
    std::string m_modelName;
    std::string m_tensorName;
    ov::PartialShape m_oldPartialShape;
    ov::PartialShape m_newPartialShape;
};

using PartialShapeParam = std::tuple<BaseFEParam, PartShape>;

class FrontEndPartialShapeTest : public ::testing::TestWithParam<PartialShapeParam> {
public:
    BaseFEParam m_baseParam;
    PartShape m_partShape;
    ov::frontend::FrontEndManager m_fem;
    ov::frontend::FrontEnd::Ptr m_frontEnd;
    ov::frontend::InputModel::Ptr m_inputModel;

    static std::string getTestCaseName(const testing::TestParamInfo<PartialShapeParam>& obj);

    void SetUp() override;

protected:
    void initParamTest();

    void doLoadFromFile();
};
