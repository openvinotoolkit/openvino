// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FLUID_TEST_COMPUTATIONS_HPP
#define FLUID_TEST_COMPUTATIONS_HPP

#include <ie_api.h>

#include <memory>
#include <vector>

namespace opencv_test
{
namespace test
{
struct Mat
{
    int   rows;
    int   cols;
    int   type;
    void* data;
};
}

class __attribute__((visibility("default"))) FluidComputation
{
protected:
    struct Priv;
    std::shared_ptr<Priv> m_priv;
public:
    FluidComputation(Priv* priv);
    void warmUp();
    void apply();
};

class __attribute__((visibility("default"))) FluidResizeComputation : public FluidComputation
{
public:
    FluidResizeComputation(test::Mat inMat, test::Mat outMat, int interp);
};

class __attribute__((visibility("default"))) FluidSplitComputation : public FluidComputation
{
public:
    FluidSplitComputation(test::Mat inMat, std::vector<test::Mat> outMats);
};

class __attribute__((visibility("default"))) FluidMergeComputation : public FluidComputation
{
public:
    FluidMergeComputation(std::vector<test::Mat> inMats, test::Mat outMat);
};

} // namespace opencv_test

#endif // FLUID_TEST_COMPUTATIONS_HPP
