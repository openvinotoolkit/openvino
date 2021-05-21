// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fluid_test_computations.hpp>
#include <opencv2/gapi.hpp>
#include <ie_preprocess_gapi_kernels.hpp>
#include <opencv2/gapi/fluid/gfluidkernel.hpp>

#define CV_MAT_CHANNELS(flags) (((flags) >> CV_CN_SHIFT) + 1)

struct FluidComputation::Priv
{
    cv::GComputation m_c;
    cv::GRunArgs m_v_in;
    std::vector<cv::gapi::own::Mat> m_v_out;

    Priv(cv::GComputation && c, std::vector<cv::gapi::own::Mat>&& v_in, std::vector<cv::gapi::own::Mat>&& v_out)
        : m_c(std::move(c)),
          m_v_in(v_in.begin(), v_in.end()),
          m_v_out(std::move(v_out))
    {}

    Priv(cv::GComputation && c, cv::gapi::own::Mat&& v_in, cv::gapi::own::Mat&& v_out)
        : m_c(std::move(c)),
          m_v_in{std::move(v_in)},
          m_v_out{std::move(v_out)}
    {}

    Priv(cv::GComputation && c, cv::gapi::own::Mat&& v_in, std::vector<cv::gapi::own::Mat>&& v_out)
        : m_c(std::move(c)),
          m_v_in{std::move(v_in)},
          m_v_out(std::move(v_out))
    {}

    Priv(cv::GComputation && c, cv::GRunArgs&& v_in, std::vector<cv::gapi::own::Mat>&& v_out)
        : m_c(std::move(c)),
          m_v_in(std::move(v_in)),
          m_v_out(std::move(v_out))
    {}

    cv::GRunArgs  ins()  { return m_v_in;}
    cv::GRunArgsP outs() {
        cv::GRunArgsP call_outs;

        for (auto &m : m_v_out) { call_outs.emplace_back(&m); }

        return call_outs;
    }

};

FluidComputation::FluidComputation(Priv *priv)
    : m_priv(priv)
{}

namespace
{
    cv::gapi::own::Rect to_own(test::Rect rect) { return {rect.x, rect.y, rect.width, rect.height}; }
}

void FluidComputation::warmUp(test::Rect roi)
{
    auto compile_args = roi.empty() ? cv::compile_args(InferenceEngine::gapi::preprocKernels())
                                    : cv::compile_args(InferenceEngine::gapi::preprocKernels(),
                                                       cv::GFluidOutputRois{{to_own(roi)}});

    m_priv->m_c.apply(m_priv->ins(), m_priv->outs(), std::move(compile_args));
}

void FluidComputation::apply()
{
    m_priv->m_c.apply(m_priv->ins(), m_priv->outs());
}

namespace
{
cv::gapi::own::Scalar to_own(test::Scalar const& s) {
    return {s.v[0], s.v[1], s.v[2], s.v[3]};
}

cv::gapi::own::Mat to_own(test::Mat mat) {
    return {mat.rows, mat.cols, mat.type, mat.data, mat.step};
}

std::vector<cv::gapi::own::Mat> to_own(std::vector<test::Mat> mats)
{
    std::vector<cv::gapi::own::Mat> own_mats(mats.size());
    for (int i = 0; i < mats.size(); i++) {
        own_mats[i] = to_own(mats[i]);
    }
    return own_mats;
}

template<typename... Ts, int... IIs>
std::vector<cv::GMat> to_vec_impl(std::tuple<Ts...> &&gmats, cv::detail::Seq<IIs...>) {
    return { std::get<IIs>(gmats)... };
}

template<typename... Ts>
std::vector<cv::GMat> to_vec(std::tuple<Ts...> &&gmats) {
    return to_vec_impl(std::move(gmats), typename cv::detail::MkSeq<sizeof...(Ts)>::type());
}
} // anonymous namespace

static cv::GComputation buildResizeComputation(test::Mat inMat, test::Mat outMat, int interp)
{
    cv::gapi::own::Size sz_in  { inMat.cols,  inMat.rows};
    cv::gapi::own::Size sz_out {outMat.cols, outMat.rows};
    int type = outMat.type;
    cv::GMat in, out;
    switch (CV_MAT_CHANNELS(type)) {
    case 1:
        out = InferenceEngine::gapi::ScalePlane::on(in, type, sz_in, sz_out, interp);
        break;
    case 3:
        {
        int depth = CV_MAT_DEPTH(type);
        int type1 = CV_MAKE_TYPE(depth, 1);
        cv::GMat in0, in1, in2, out0, out1, out2;
        std::tie(in0, in1, in2) = InferenceEngine::gapi::Split3::on(in);
        out0 = InferenceEngine::gapi::ScalePlane::on(in0, type1, sz_in, sz_out, interp);
        out1 = InferenceEngine::gapi::ScalePlane::on(in1, type1, sz_in, sz_out, interp);
        out2 = InferenceEngine::gapi::ScalePlane::on(in2, type1, sz_in, sz_out, interp);
        out = InferenceEngine::gapi::Merge3::on(out0, out1, out2);
        }
        break;
    default: GAPI_Assert(!"ERROR: unsupported number of channels!");
    }

    return cv::GComputation(in, out);
}

FluidResizeComputation::FluidResizeComputation(test::Mat inMat, test::Mat outMat, int interp)
    : FluidComputation(new Priv{buildResizeComputation(inMat, outMat, interp)
                               ,to_own(inMat)
                               ,to_own(outMat)
                               })
{}

static cv::GComputation buildResizeRGB8UComputation(test::Mat inMat, test::Mat outMat, int interp)
{
    cv::gapi::own::Size sz_in  { inMat.cols,  inMat.rows};
    cv::gapi::own::Size sz_out {outMat.cols, outMat.rows};
    int type = outMat.type;
    cv::GMat in, out, out_r, out_g, out_b, out_x;

    if (type == CV_8UC3) {
        std::tie(out_r, out_g, out_b) = InferenceEngine::gapi::ScalePlanes::on(in, type, sz_in, sz_out, interp);
        out = InferenceEngine::gapi::Merge3::on(out_r, out_g, out_b);
    }
    else if (type == CV_8UC4) {
        std::tie(out_r, out_g, out_b, out_x) = InferenceEngine::gapi::ScalePlanes4::on(in, type, sz_in, sz_out, interp);
        out = InferenceEngine::gapi::Merge4::on(out_r, out_g, out_b, out_x);
    } else {
        GAPI_Assert(!"ERROR: unsupported number of channels!");
    }

    return cv::GComputation(in, out);
}

FluidResizeRGB8UComputation::FluidResizeRGB8UComputation(test::Mat inMat, test::Mat outMat, int interp)
    : FluidComputation(new Priv{buildResizeRGB8UComputation(inMat, outMat, interp)
                               ,to_own(inMat)
                               ,to_own(outMat)
                               })
{}

static cv::GComputation buildSplitComputation(int planes)
{
    std::vector<cv::GMat> ins(1);
    std::vector<cv::GMat> outs(planes);

    switch (planes) {
    case 2: outs = to_vec(InferenceEngine::gapi::Split2::on(ins[0])); break;
    case 3: outs = to_vec(InferenceEngine::gapi::Split3::on(ins[0])); break;
    case 4: outs = to_vec(InferenceEngine::gapi::Split4::on(ins[0])); break;
    default: GAPI_Assert(false);
    }

    return cv::GComputation(ins, outs);
}

FluidSplitComputation::FluidSplitComputation(test::Mat inMat, std::vector<test::Mat> outMats)
    : FluidComputation(new Priv{buildSplitComputation(outMats.size())
                               ,to_own(inMat)
                               ,to_own(outMats)
                               })
{}

static cv::GComputation buildChanToPlaneComputation(int chan)
{
    cv::GMat in, out;
    out = InferenceEngine::gapi::ChanToPlane::on(in, chan);
    return cv::GComputation(in, out);
}

FluidChanToPlaneComputation::FluidChanToPlaneComputation(test::Mat inMat, test::Mat outMat, int chan)
    : FluidComputation(new Priv{buildChanToPlaneComputation(chan)
                               ,to_own(inMat)
                               ,to_own(outMat)
                               })
{}

static cv::GComputation buildMergeComputation(int planes)
{
    std::vector<cv::GMat> ins(planes);
    std::vector<cv::GMat> outs(1);

    switch (planes) {
    case 2: outs[0] = InferenceEngine::gapi::Merge2::on(ins[0], ins[1]); break;
    case 3: outs[0] = InferenceEngine::gapi::Merge3::on(ins[0], ins[1], ins[2]); break;
    case 4: outs[0] = InferenceEngine::gapi::Merge4::on(ins[0], ins[1], ins[2], ins[3]); break;
    default: GAPI_Assert(false);
    }

    return cv::GComputation(ins, outs);
}

FluidMergeComputation::FluidMergeComputation(std::vector<test::Mat> inMats, test::Mat outMat)
    : FluidComputation(new Priv{buildMergeComputation(inMats.size())
                               ,to_own(inMats)
                               ,{to_own(outMat)}
                               })
{}

static cv::GComputation buildFluidNV12toRGBComputation()
{
    cv::GMat in_y, in_uv;
    cv::GMat out = InferenceEngine::gapi::NV12toRGB::on(in_y,in_uv);
    return cv::GComputation(cv::GIn(in_y,in_uv), cv::GOut(out));
}

FluidNV12toRGBComputation::FluidNV12toRGBComputation(test::Mat inMat_y, test::Mat inMat_uv, test::Mat outMat)
    : FluidComputation(new Priv{buildFluidNV12toRGBComputation()
                               ,to_own({inMat_y,inMat_uv})
                               ,{to_own(outMat)}
                               })
{}

static cv::GComputation buildFluidI420toRGBComputation()
{
    cv::GMat in_y, in_u, in_v;
    cv::GMat out = InferenceEngine::gapi::I420toRGB::on(in_y, in_u, in_v);
    return cv::GComputation(cv::GIn(in_y, in_u, in_v), cv::GOut(out));
}


FluidI420toRGBComputation::FluidI420toRGBComputation(test::Mat inMat_y, test::Mat inMat_u, test::Mat inMat_v, test::Mat outMat)
    : FluidComputation(new Priv{buildFluidI420toRGBComputation()
                               ,to_own({inMat_y,inMat_u, inMat_v})
                               ,{to_own(outMat)}
                               })
{}

ConvertDepthComputation::ConvertDepthComputation(test::Mat inMat, test::Mat outMat,  int depth)
    : FluidComputation(new Priv{ [depth]()-> cv::GComputation {
                                      cv::GMat in;
                                      cv::GMat out = InferenceEngine::gapi::ConvertDepth::on(in, depth);
                                      return cv::GComputation(cv::GIn(in), cv::GOut(out));
                                  }()
                               , to_own(inMat)
                               , to_own(outMat)
                               })
{}

DivCComputation::DivCComputation(test::Mat inMat, test::Mat outMat, test::Scalar const& c)
    : FluidComputation(new Priv{ []()-> cv::GComputation {
                                      cv::GMat in;
                                      cv::GScalar C;
                                      cv::GMat out = in / C;
                                      return cv::GComputation(cv::GIn(in, C), cv::GOut(out));
                                  }()
                                , cv::GRunArgs{cv::GRunArg{to_own(inMat)}, cv::GRunArg{to_own(c)}}
                                , {to_own(outMat)}
                               })
{}

SubCComputation::SubCComputation(test::Mat inMat, test::Mat outMat, test::Scalar const& c)
    : FluidComputation(new Priv{ []()-> cv::GComputation{
                                      cv::GMat in;
                                      cv::GScalar C;
                                      cv::GMat out = in - C;
                                      return cv::GComputation(cv::GIn(in, C), cv::GOut(out));
                                  }()
                                , cv::GRunArgs{cv::GRunArg{to_own(inMat)}, cv::GRunArg{to_own(c)}}
                                , {to_own(outMat)}
                               })
{}

MeanValueSubtractComputation::MeanValueSubtractComputation(test::Mat inMat, test::Mat outMat, test::Scalar const& mean, test::Scalar const& std)
    : FluidComputation(new Priv{ []()-> cv::GComputation{
                                      cv::GMat in;
                                      cv::GScalar _mean;
                                      cv::GScalar _std;
                                      cv::GMat out = (in - _mean) / _std;
                                      return cv::GComputation(cv::GIn(in, _mean, _std), cv::GOut(out));
                                  }()
                                , cv::GRunArgs{cv::GRunArg{to_own(inMat)}, cv::GRunArg{to_own(mean)}, cv::GRunArg{to_own(std)}}
                                , {to_own(outMat)}
                               })
{}

namespace cv {
cv::GMat operator-(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return InferenceEngine::gapi::GSubC::on(lhs, rhs, -1);
}
cv::GMat operator/(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return InferenceEngine::gapi::GDivC::on(lhs, rhs, 1.0, -1);
}

}
