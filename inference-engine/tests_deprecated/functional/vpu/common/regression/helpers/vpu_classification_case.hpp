// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tests_common.hpp>
#include "vpu_case_common.hpp"

using CompilationTestParam = WithParamInterface<std::tuple<PluginDevicePair, CompilationParameter>>;
using ClassificationTestVpuParam = WithParamInterface<std::tuple<
        PluginDevicePair,
        Precision,
        Batch,
        DoReshape,
        Resources,
        ClassificationSrcParam>>;

using ClassificationSpecificTestVpuParam = WithParamInterface<std::tuple<
        PluginDevicePair,
        Precision,
        Batch,
        DoReshape>>;

//------------------------------------------------------------------------------
// class VpuNoClassificationRegression
//------------------------------------------------------------------------------

class VpuNoClassificationRegression : public VpuNoRegressionBase,
                                      public ClassificationTestVpuParam {
public:
    //Operations
    static std::string getTestCaseName(
            TestParamInfo<ClassificationTestVpuParam::ParamType> param);

protected:
    // Data section
    int resources_;
    ClassificationSrcParam source_param_;

    //Operations
    void SetUp() override;
    void InitConfig() override;
};

//------------------------------------------------------------------------------
// class VpuNoClassificationRegressionSpecific
//------------------------------------------------------------------------------

class VpuNoClassificationRegressionSpecific : public VpuNoRegressionBase,
                                              public ClassificationSpecificTestVpuParam {
public:
    //Operations
    static std::string getTestCaseName(
            TestParamInfo<ClassificationSpecificTestVpuParam::ParamType> param);

protected:
    //Operations
    void SetUp() override;
    void InitConfig() override;
};

//------------------------------------------------------------------------------
// class VpuNoRegressionWithCompilation
//------------------------------------------------------------------------------

class VpuNoRegressionWithCompilation : public TestsCommon,
                                       public CompilationTestParam {
public:
    // Operations
    static std::string getTestCaseName(TestParamInfo <CompilationTestParam::ParamType> param);

    // Accessors
    std::string getDeviceName() const;

protected:
    // Data section
    std::string plugin_name_;
    std::string device_name_;
    CompilationParameter compilation_param_;

    //Operations
    void SetUp() override;
};

//------------------------------------------------------------------------------
// Implementation of methods of class VpuNoRegressionWithCompilation
//------------------------------------------------------------------------------

inline std::string VpuNoRegressionWithCompilation::getDeviceName() const {
    return plugin_name_;
}
