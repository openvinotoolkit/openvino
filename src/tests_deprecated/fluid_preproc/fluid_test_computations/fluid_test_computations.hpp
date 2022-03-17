// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FLUID_TEST_COMPUTATIONS_HPP
#define FLUID_TEST_COMPUTATIONS_HPP

#include <ie_api.h>

#include <memory>
#include <vector>
#include <array>

namespace test
{
struct Mat
{
    int     rows;
    int     cols;
    int     type;
    void*   data;
    size_t  step;
};
struct Rect{
    int x;
    int y;
    int width;
    int height;
    bool empty(){
        return width == 0 && height == 0;
    };
};
struct Scalar
{
    std::array<double, 4> v;
};

}

class FluidComputation
{
protected:
    struct Priv;
    std::shared_ptr<Priv> m_priv;
public:
    FluidComputation(Priv* priv);
    void warmUp(test::Rect roi = {});
    void apply();
};

class FluidResizeComputation : public FluidComputation
{
public:
    FluidResizeComputation(test::Mat inMat, test::Mat outMat, int interp);
};

class FluidResizeRGB8UComputation : public FluidComputation
{
public:
    FluidResizeRGB8UComputation(test::Mat inMat, test::Mat outMat, int interp);
};

class FluidSplitComputation : public FluidComputation
{
public:
    FluidSplitComputation(test::Mat inMat, std::vector<test::Mat> outMats);
};

class FluidChanToPlaneComputation : public FluidComputation
{
public:
    FluidChanToPlaneComputation(test::Mat inMat, test::Mat outMat, int chan);
};

class FluidMergeComputation : public FluidComputation
{
public:
    FluidMergeComputation(std::vector<test::Mat> inMats, test::Mat outMat);
};

class FluidNV12toRGBComputation : public FluidComputation
{
public:
    FluidNV12toRGBComputation(test::Mat inMat_y, test::Mat inMat_uv, test::Mat outMat);
};

class FluidI420toRGBComputation : public FluidComputation
{
public:
    FluidI420toRGBComputation(test::Mat inMat_y, test::Mat inMat_u, test::Mat inMat_v, test::Mat outMat);
};

class ConvertDepthComputation : public FluidComputation
{
public:
    ConvertDepthComputation(test::Mat inMat, test::Mat outMat, int depth);
};

class DivCComputation : public FluidComputation
{
public:
    DivCComputation(test::Mat inMat, test::Mat outMat, test::Scalar const& c);
};

class SubCComputation : public FluidComputation
{
public:
    SubCComputation(test::Mat inMat, test::Mat outMat, test::Scalar const& c);
};

class MeanValueSubtractComputation : public FluidComputation
{
public:
    MeanValueSubtractComputation(test::Mat inMat, test::Mat outMat, test::Scalar const& mean, test::Scalar const& std);
};

#endif // FLUID_TEST_COMPUTATIONS_HPP
