// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fluid_test_computations.hpp>
#include <opencv2/gapi.hpp>
#include <ie_preprocess_gapi_kernels.hpp>

#define CV_MAT_CHANNELS(flags) (((flags) >> CV_CN_SHIFT) + 1)

namespace opencv_test
{
struct FluidComputation::Priv
{
    cv::GComputation m_c;
    std::vector<cv::gapi::own::Mat> m_v_in;
    std::vector<cv::gapi::own::Mat> m_v_out;
};

FluidComputation::FluidComputation(Priv *priv)
    : m_priv(priv)
{}

void FluidComputation::warmUp()
{
    m_priv->m_c.apply(m_priv->m_v_in, m_priv->m_v_out, cv::compile_args(InferenceEngine::gapi::preprocKernels()));
}

void FluidComputation::apply()
{
    m_priv->m_c.apply(m_priv->m_v_in, m_priv->m_v_out);
}

namespace
{
cv::gapi::own::Mat to_own(test::Mat mat) { return {mat.rows, mat.cols, mat.type, mat.data}; }

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
                               ,{to_own(inMat)}
                               ,{to_own(outMat)}
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
                               ,{to_own(inMat)}
                               ,to_own(outMats)
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

} // namespace opencv_test
