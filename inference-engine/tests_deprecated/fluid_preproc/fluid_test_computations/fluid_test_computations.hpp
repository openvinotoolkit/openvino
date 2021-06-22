// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FLUID_TEST_COMPUTATIONS_HPP
#define FLUID_TEST_COMPUTATIONS_HPP

#include <ie_api.h>

#include <memory>
#include <vector>
#include <array>

#if defined(_WIN32)
    #ifdef IMPLEMENT_FLUID_COMPUTATION_API
        #define FLUID_COMPUTATION_VISIBILITY __declspec(dllexport)
    #else
        #define FLUID_COMPUTATION_VISIBILITY __declspec(dllimport)
    #endif
#else
    #ifdef IMPLEMENT_FLUID_COMPUTATION_API
        #define FLUID_COMPUTATION_VISIBILITY __attribute__((visibility("default")))
    #else
        #define FLUID_COMPUTATION_VISIBILITY
    #endif
#endif

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

class FLUID_COMPUTATION_VISIBILITY FluidComputation
{
protected:
    struct Priv;
    std::shared_ptr<Priv> m_priv;
public:
    FluidComputation(Priv* priv);
    void warmUp(test::Rect roi = {});
    void apply();
};

class FLUID_COMPUTATION_VISIBILITY FluidResizeComputation : public FluidComputation
{
public:
    FluidResizeComputation(test::Mat inMat, test::Mat outMat, int interp);
};

class FLUID_COMPUTATION_VISIBILITY FluidResizeRGB8UComputation : public FluidComputation
{
public:
    FluidResizeRGB8UComputation(test::Mat inMat, test::Mat outMat, int interp);
};

class FLUID_COMPUTATION_VISIBILITY FluidSplitComputation : public FluidComputation
{
public:
    FluidSplitComputation(test::Mat inMat, std::vector<test::Mat> outMats);
};

class FLUID_COMPUTATION_VISIBILITY FluidChanToPlaneComputation : public FluidComputation
{
public:
    FluidChanToPlaneComputation(test::Mat inMat, test::Mat outMat, int chan);
};

class FLUID_COMPUTATION_VISIBILITY FluidMergeComputation : public FluidComputation
{
public:
    FluidMergeComputation(std::vector<test::Mat> inMats, test::Mat outMat);
};

class FLUID_COMPUTATION_VISIBILITY FluidNV12toRGBComputation : public FluidComputation
{
public:
    FluidNV12toRGBComputation(test::Mat inMat_y, test::Mat inMat_uv, test::Mat outMat);
};

class FLUID_COMPUTATION_VISIBILITY FluidI420toRGBComputation : public FluidComputation
{
public:
    FluidI420toRGBComputation(test::Mat inMat_y, test::Mat inMat_u, test::Mat inMat_v, test::Mat outMat);
};

class FLUID_COMPUTATION_VISIBILITY ConvertDepthComputation : public FluidComputation
{
public:
    ConvertDepthComputation(test::Mat inMat, test::Mat outMat, int depth);
};

class FLUID_COMPUTATION_VISIBILITY DivCComputation : public FluidComputation
{
public:
    DivCComputation(test::Mat inMat, test::Mat outMat, test::Scalar const& c);
};

class FLUID_COMPUTATION_VISIBILITY SubCComputation : public FluidComputation
{
public:
    SubCComputation(test::Mat inMat, test::Mat outMat, test::Scalar const& c);
};

class FLUID_COMPUTATION_VISIBILITY MeanValueSubtractComputation : public FluidComputation
{
public:
    MeanValueSubtractComputation(test::Mat inMat, test::Mat outMat, test::Scalar const& mean, test::Scalar const& std);
};

#endif // FLUID_TEST_COMPUTATIONS_HPP
