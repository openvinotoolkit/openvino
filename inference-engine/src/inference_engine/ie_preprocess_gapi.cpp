// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <vector>
#include <algorithm>
#include <tuple>
#include <string>

// Careful reader, don't worry -- it is not the whole OpenCV,
// it is just a single stand-alone component of it
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/util/util.hpp>

#include "ie_blob.h"
#include "ie_input_info.hpp"
#include "ie_preprocess_gapi.hpp"
#include "ie_preprocess_gapi_kernels.hpp"

#include "ie_parallel.hpp"

#include <opencv2/gapi/fluid/gfluidkernel.hpp>  // GFluidOutputRois

namespace InferenceEngine {
namespace {
namespace G {
    struct Strides {int N; int C; int H; int W;};
    struct Dims    {int N; int C; int H; int W;};
    struct Desc    {Dims d; Strides s;};

    void fix_strides_nhwc(const Dims &d, Strides &s) {
        if (s.W > d.C) {
            s.C = 1;
            s.W = s.C*d.C;
            s.H = s.W*d.W;
            s.N = s.H*d.H;
        }
    }

    Desc decompose(Blob::Ptr &blob) {
        const auto& ie_desc     = blob->getTensorDesc();
        const auto& ie_blk_desc = ie_desc.getBlockingDesc();
        const auto& ie_dims     = ie_desc.getDims();
        const auto& ie_strides  = ie_blk_desc.getStrides();

        Dims d = {
            static_cast<int>(ie_dims[0]),
            static_cast<int>(ie_dims[1]),
            static_cast<int>(ie_dims[2]),
            static_cast<int>(ie_dims[3])
        };

        Strides s = {
            static_cast<int>(ie_strides[0]),
            static_cast<int>(blob->layout() == NHWC ? ie_strides[3] : ie_strides[1]),
            static_cast<int>(blob->layout() == NHWC ? ie_strides[1] : ie_strides[2]),
            static_cast<int>(blob->layout() == NHWC ? ie_strides[2] : ie_strides[3]),
        };

        if (blob->layout() == NHWC) fix_strides_nhwc(d, s);

        return Desc{d, s};
    }
}  // namespace G

inline int get_cv_depth(const InferenceEngine::TensorDesc &ie_desc) {
    switch (ie_desc.getPrecision()) {
    case Precision::U8:   return CV_8U;
    case Precision::FP32: return CV_32F;
    default: THROW_IE_EXCEPTION << "Unsupported data type";
    }
}

std::vector<cv::gapi::own::Mat> bind_to_blob(Blob::Ptr &blob) {
    const auto& ie_desc     = blob->getTensorDesc();
    const auto& ie_desc_blk = ie_desc.getBlockingDesc();
    const auto     desc     = G::decompose(blob);
    const auto cv_depth     = get_cv_depth(ie_desc);
    const auto stride       = desc.s.H*blob->element_size();
    const auto planeSize    = cv::gapi::own::Size(desc.d.W, desc.d.H);


    uint8_t* ptr = static_cast<uint8_t*>(blob->buffer());
    ptr += blob->element_size()*ie_desc_blk.getOffsetPadding();

    std::vector<cv::gapi::own::Mat> result;
    if (blob->layout() == NHWC) {
        result.emplace_back(planeSize.height, planeSize.width, CV_MAKETYPE(cv_depth, desc.d.C), ptr, stride);
    } else {  // NCHW
        const auto planeType = CV_MAKETYPE(cv_depth, 1);
        for (size_t ch = 0; ch < desc.d.C; ch++) {
            cv::gapi::own::Mat plane(planeSize.height, planeSize.width, planeType, ptr + ch*desc.s.C*blob->element_size(), stride);
            result.emplace_back(plane);
        }
    }
    return result;
}

template<typename... Ts, int... IIs>
std::vector<cv::GMat> to_vec_impl(std::tuple<Ts...> &&gmats, cv::detail::Seq<IIs...>) {
    return { std::get<IIs>(gmats)... };
}

template<typename... Ts>
std::vector<cv::GMat> to_vec(std::tuple<Ts...> &&gmats) {
    return to_vec_impl(std::move(gmats), typename cv::detail::MkSeq<sizeof...(Ts)>::type());
}

cv::GComputation buildGraph(const G::Desc &in_desc,
                            const G::Desc &out_desc,
                            InferenceEngine::Layout in_layout,
                            InferenceEngine::Layout out_layout,
                            InferenceEngine::ResizeAlgorithm algorithm,
                            int precision) {
    if ((in_layout == NHWC) && (in_desc.d.C == 3) && (precision == CV_8U) && (algorithm == RESIZE_BILINEAR)) {
        const auto input_sz = cv::gapi::own::Size(in_desc.d.W, in_desc.d.H);
        const auto scale_sz = cv::gapi::own::Size(out_desc.d.W, out_desc.d.H);
        std::vector<cv::GMat> inputs(1);
        std::vector<cv::GMat> outputs;

        if (out_layout == NHWC) {
            outputs.resize(1);
            auto planes = to_vec(gapi::ScalePlanes::on(inputs[0], precision, input_sz, scale_sz, cv::INTER_LINEAR));
            outputs[0] = gapi::Merge3::on(planes[0], planes[1], planes[2]);
        } else {
            outputs = to_vec(gapi::ScalePlanes::on(inputs[0], precision, input_sz, scale_sz, cv::INTER_LINEAR));
        }
        return cv::GComputation(inputs, outputs);
    }

    std::vector<cv::GMat> inputs;  // 1 element if NHWC, C elements if NCHW
    std::vector<cv::GMat> planes;

    // Convert input blob to planar format, if it is not yet planar
    if (in_layout == NHWC) {
        // interleaved input blob needs to be decomposed into distinct planes
        inputs.resize(1);
        switch (in_desc.d.C) {
        case 1: planes = { inputs[0] };                       break;
        case 2: planes = to_vec(gapi::Split2::on(inputs[0])); break;
        case 3: planes = to_vec(gapi::Split3::on(inputs[0])); break;
        case 4: planes = to_vec(gapi::Split4::on(inputs[0])); break;
        default:
            for (int chan = 0; chan < in_desc.d.C; chan++)
                planes.emplace_back(gapi::ChanToPlane::on(inputs[0], chan));
            break;
        }
    } else if (in_layout == NCHW) {
        // planar blob can be passed to resize as-is
        inputs.resize(in_desc.d.C);
        planes = inputs;
    }

    // Resize every plane
    std::vector<cv::GMat> out_planes;
    const int interp_type = [](const ResizeAlgorithm &ar) {
        switch (ar) {
        case RESIZE_AREA:     return cv::INTER_AREA;
        case RESIZE_BILINEAR: return cv::INTER_LINEAR;
        default: THROW_IE_EXCEPTION << "Unsupported resize operation";
        }
    } (algorithm);
    const auto input_sz  = cv::gapi::own::Size(in_desc.d.W, in_desc.d.H);
    const auto scale_sz  = cv::gapi::own::Size(out_desc.d.W, out_desc.d.H);
    const auto scale_fcn = std::bind(&gapi::ScalePlane::on,
                                     std::placeholders::_1,
                                     precision,
                                     input_sz, scale_sz, interp_type);
    std::transform(planes.begin(), planes.end(), std::back_inserter(out_planes), scale_fcn);

    // Convert to expected layout, if required
    std::vector<cv::GMat> outputs;  // 1 element if NHWC, C elements if NCHW
    if (out_layout == NHWC) {
        outputs.resize(1);
        if      (out_desc.d.C == 1) outputs[0] = out_planes[0];
        else if (out_desc.d.C == 2) outputs[0] = gapi::Merge2::on(out_planes[0], out_planes[1]);
        else if (out_desc.d.C == 3) outputs[0] = gapi::Merge3::on(out_planes[0], out_planes[1], out_planes[2]);
        else if (out_desc.d.C == 4) outputs[0] = gapi::Merge4::on(out_planes[0], out_planes[1], out_planes[2], out_planes[3]);
        else    THROW_IE_EXCEPTION << "Output channels >4 are not supported for HWC [by G-API]";
    } else {
        outputs = out_planes;
    }

    return cv::GComputation(inputs, outputs);
}
}  // anonymous namespace

InferenceEngine::PreprocEngine::PreprocEngine() : _lastComp(parallel_get_max_threads()) {}

InferenceEngine::PreprocEngine::Update InferenceEngine::PreprocEngine::needUpdate(const CallDesc &newCallOrig) const {
    // Given our knowledge about Fluid, full graph rebuild is required
    // if and only if:
    // 0. This is the first call ever
    // 1. precision has changed (affects kernel versions)
    // 2. layout has changed (affects graph topology)
    // 3. algorithm has changed (affects kernel version)
    // 4. dimensions have changed from downscale to upscale or
    // vice-versa if interpolation is AREA.
    if (!_lastCall) {
        return Update::REBUILD;
    }

    BlobDesc last_in;
    BlobDesc last_out;
    ResizeAlgorithm last_algo;
    std::tie(last_in, last_out, last_algo) = *_lastCall;

    CallDesc newCall = newCallOrig;
    BlobDesc new_in;
    BlobDesc new_out;
    ResizeAlgorithm new_algo;
    std::tie(new_in, new_out, new_algo) = newCall;

    // Declare two empty vectors per each call
    SizeVector last_in_size;
    SizeVector last_out_size;
    SizeVector new_in_size;
    SizeVector new_out_size;

    // Now swap it with in/out descriptor vectors
    // Now last_in/last_out would contain everything but sizes
    last_in_size.swap(std::get<2>(last_in));
    last_out_size.swap(std::get<2>(last_out));
    new_in_size.swap(std::get<2>(new_in));
    new_out_size.swap(std::get<2>(new_out));

    // If anything (except input sizes) changes, rebuild is required
    if (last_in != new_in || last_out != new_out || last_algo != new_algo) {
        return Update::REBUILD;
    }

    // If output sizes change, graph should be regenerated (resize
    // ratio is taken from parameters)
    if (last_out_size != new_out_size) {
        return Update::REBUILD;
    }

    // If interpolation is AREA and sizes change upscale/downscale
    // mode, rebuild is required
    if (last_algo == RESIZE_AREA) {
        // 0123 == NCHW
        const auto is_upscale = [](const SizeVector &in, const SizeVector &out) -> bool {
            return in[2] < out[2] || in[3] < out[3];
        };
        const bool old_upscale = is_upscale(last_in_size, last_out_size);
        const bool new_upscale = is_upscale(new_in_size, new_out_size);
        if (old_upscale != new_upscale) {
            return Update::REBUILD;
        }
    }

    // If only sizes changes (considering the above exception),
    // reshape is enough
    if (last_in_size != new_in_size) {
        return Update::RESHAPE;
    }

    return Update::NOTHING;
}

bool InferenceEngine::PreprocEngine::preprocessWithGAPI(Blob::Ptr &inBlob, Blob::Ptr &outBlob, const ResizeAlgorithm &algorithm, bool omp_serial) {
    static const bool NO_GAPI = [](const char *str) -> bool {
        std::string var(str ? str : "");
        return var == "N" || var == "NO" || var == "OFF" || var == "0";
    } (std::getenv("USE_GAPI"));

    if (NO_GAPI)
        return false;

    const auto &in_desc_ie = inBlob->getTensorDesc();
    const auto &out_desc_ie = outBlob->getTensorDesc();
    auto supports_layout = [](Layout l) { return l == Layout::NCHW || l == Layout::NHWC; };
    if (!supports_layout(inBlob->layout()) || !supports_layout(outBlob->layout())
        || in_desc_ie.getDims().size() != 4 || out_desc_ie.getDims().size() != 4) {
        THROW_IE_EXCEPTION << "Preprocess support NCHW/NHWC only";
    }

    const G::Desc
        in_desc = G::decompose(inBlob),
        out_desc = G::decompose(outBlob);

    CallDesc thisCall = CallDesc{ BlobDesc{ in_desc_ie.getPrecision(),
                                            inBlob->layout(),
                                            in_desc_ie.getDims() },
                                  BlobDesc{ out_desc_ie.getPrecision(),
                                            outBlob->layout(),
                                            out_desc_ie.getDims() },
                                  algorithm };
    const Update update = needUpdate(thisCall);

    std::vector<cv::gapi::own::Mat> input_plane_mats  = bind_to_blob(inBlob);
    std::vector<cv::gapi::own::Mat> output_plane_mats = bind_to_blob(outBlob);

    Opt<cv::GComputation> _lastComputation;
    if (Update::REBUILD == update || Update::RESHAPE == update) {
        _lastCall = cv::util::make_optional(std::move(thisCall));

        if (Update::REBUILD == update) {
            //  rebuild the graph
            IE_PROFILING_AUTO_SCOPE_TASK(_perf_graph_building);
            _lastComputation = cv::util::make_optional(buildGraph(in_desc,
                                                                  out_desc,
                                                                  inBlob->layout(),
                                                                  outBlob->layout(),
                                                                  algorithm,
                                                                  get_cv_depth(in_desc_ie)));
        }
    }

    const int thread_num =
            #if IE_THREAD == IE_THREAD_OMP
                omp_serial ? 1 :    // disable threading for OpenMP if was asked for
            #endif
                0;                  // use all available threads

    // to suppress unused warnings
    (void)(omp_serial);

    // Split the whole graph into `total_slices` slices, where
    // `total_slices` is provided by the parallel runtime and assumed
    // to be number of threads used.  However it is not guaranteed
    // that an actual number of threads will be as assumed, so it
    // possible that all slices are processed by the same thread.
    //
    parallel_nt_static(thread_num , [&, this](int slice_n, const int total_slices){
        IE_PROFILING_AUTO_SCOPE_TASK(_perf_exec_tile);

        auto& compiled = _lastComp[slice_n];
        if (Update::REBUILD == update || Update::RESHAPE == update) {
            //  need to compile (or reshape) own object for a particular ROI
            IE_PROFILING_AUTO_SCOPE_TASK(_perf_graph_compiling);

            auto meta_of = [](std::vector<cv::gapi::own::Mat> const& ins){
                std::vector<cv::GMetaArg> rslt{ins.size()}; rslt.clear();
                for (auto& m : ins) {
                    rslt.emplace_back(descr_of(m));
                }
                return rslt;
            };

            using cv::gapi::own::Rect;

            const auto lines_per_thread = output_plane_mats[0].rows / total_slices;
            const auto remainder = output_plane_mats[0].rows - total_slices * lines_per_thread;
            const auto roi_height = lines_per_thread + ((slice_n == total_slices -1) ?  remainder : 0);

            auto roi = Rect{0, slice_n * lines_per_thread, output_plane_mats[0].cols, roi_height};
            std::vector<Rect> rois(output_plane_mats.size(), roi);

            // TODO: make a ROI a runtime argument to avoid
            // recompilations
            auto args = cv::compile_args(gapi::preprocKernels(), cv::GFluidOutputRois{std::move(rois)});
            if (Update::REBUILD == update) {
                auto& computation = _lastComputation.value();
                compiled = computation.compile(meta_of(input_plane_mats), std::move(args));
            } else {
                IE_ASSERT(compiled);
                compiled.reshape(meta_of(input_plane_mats), std::move(args));
            }
        }

        cv::GRunArgs call_ins;
        cv::GRunArgsP call_outs;
        for (const auto & m : input_plane_mats) { call_ins.emplace_back(m);}
        for (auto & m : output_plane_mats) { call_outs.emplace_back(&m);}

        IE_PROFILING_AUTO_SCOPE_TASK(_perf_exec_graph);
        compiled(std::move(call_ins), std::move(call_outs));
    });

    return true;
}
}  // namespace InferenceEngine
