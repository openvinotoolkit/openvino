// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <tuple>
#include <string>
#include <unordered_map>
#include <functional>

// Careful reader, don't worry -- it is not the whole OpenCV,
// it is just a single stand-alone component of it
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/util/util.hpp>

#include "ie_blob.h"
#include "ie_compound_blob.h"
#include "ie_input_info.hpp"
#include "ie_preprocess_gapi.hpp"
#include "ie_preprocess_gapi_kernels.hpp"
#include "ie_preprocess_itt.hpp"
#include "debug.h"

#include "ie_parallel.hpp"

#include <opencv2/gapi/fluid/gfluidkernel.hpp>  // GFluidOutputRois

IE_SUPPRESS_DEPRECATED_START

namespace InferenceEngine {
namespace {
int get_cv_depth(const TensorDesc &ie_desc);

namespace G {
    struct Strides {int N; int C; int H; int W;};
    struct Dims    {int N; int C; int H; int W;};
    struct Desc    {Dims d; Strides s; int prec;};

    void fix_strides_nhwc(const Dims &d, Strides &s) {
        if (s.W > d.C) {
            s.C = 1;
            s.W = s.C*d.C;
            s.H = s.W*d.W;
            s.N = s.H*d.H;
        }
    }

    Desc decompose(const TensorDesc& ie_desc) {
        const auto& ie_blk_desc = ie_desc.getBlockingDesc();
        const auto& ie_dims     = ie_desc.getDims();
        const auto& ie_strides  = ie_blk_desc.getStrides();
        const bool  nhwc_layout = ie_desc.getLayout() == NHWC;

        Dims d = {
            static_cast<int>(ie_dims[0]),
            static_cast<int>(ie_dims[1]),
            static_cast<int>(ie_dims[2]),
            static_cast<int>(ie_dims[3])
        };

        Strides s = {
            static_cast<int>(ie_strides[0]),
            static_cast<int>(nhwc_layout ? ie_strides[3] : ie_strides[1]),
            static_cast<int>(nhwc_layout ? ie_strides[1] : ie_strides[2]),
            static_cast<int>(nhwc_layout ? ie_strides[2] : ie_strides[3]),
        };

        if (nhwc_layout) fix_strides_nhwc(d, s);

        return Desc{d, s, get_cv_depth(ie_desc)};
    }

    Desc decompose(const Blob::Ptr& blob) {
        return decompose(blob->getTensorDesc());
    }
}  // namespace G

inline int get_cv_depth(const TensorDesc &ie_desc) {
    switch (ie_desc.getPrecision()) {
    case Precision::U8:   return CV_8U;
    case Precision::FP32: return CV_32F;
    case Precision::U16:  return CV_16U;
    case Precision::I16:  return CV_16S;
    case Precision::FP16: return CV_16F;

    default: IE_THROW() << "Unsupported data type";
    }
}

std::vector<std::vector<cv::gapi::own::Mat>> bind_to_blob(const Blob::Ptr& blob,
                                                          int batch_size) {
    const auto& ie_desc     = blob->getTensorDesc();
    const auto& ie_desc_blk = ie_desc.getBlockingDesc();
    const auto     desc     = G::decompose(blob);
    const auto cv_depth     = get_cv_depth(ie_desc);
    const auto stride       = desc.s.H*blob->element_size();
    const auto planeSize    = cv::gapi::own::Size(desc.d.W, desc.d.H);
    // Note: operating with strides (desc.s) rather than dimensions (desc.d) which is vital for ROI
    //       blobs (data buffer is shared but dimensions are different due to ROI != original image)
    const auto batch_offset = desc.s.N * blob->element_size();

    std::vector<std::vector<cv::gapi::own::Mat>> result(batch_size);

    uint8_t* blob_ptr = static_cast<uint8_t*>(blob->buffer());
    if (blob_ptr == nullptr) {
        IE_THROW() << "Blob buffer is nullptr";
    }
    blob_ptr += blob->element_size()*ie_desc_blk.getOffsetPadding();

    for (int i = 0; i < batch_size; ++i) {
        uint8_t* curr_data_ptr = blob_ptr + i * batch_offset;

        std::vector<cv::gapi::own::Mat> planes;
        if (ie_desc.getLayout() == Layout::NHWC) {
            planes.emplace_back(planeSize.height, planeSize.width, CV_MAKETYPE(cv_depth, desc.d.C),
                curr_data_ptr, stride);
        } else {  // NCHW
            if (desc.d.C <= 0) {
                IE_THROW() << "Invalid number of channels in blob tensor descriptor, "
                                      "expected >0, actual: " << desc.d.C;
            }
            const auto planeType = CV_MAKETYPE(cv_depth, 1);
            for (int ch = 0; ch < desc.d.C; ch++) {
                cv::gapi::own::Mat plane(planeSize.height, planeSize.width, planeType,
                    curr_data_ptr + ch*desc.s.C*blob->element_size(), stride);
                planes.emplace_back(plane);
            }
        }

        result[i] = std::move(planes);
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

// convert input to planar format
std::vector<cv::GMat> split(const std::vector<cv::GMat>& inputs,
                            int channels) {
    if (inputs.empty()) {
        return inputs;
    }

    std::vector<cv::GMat> planes;

    switch (channels) {
    case 1: planes = { inputs[0] };                       break;
    case 2: planes = to_vec(gapi::Split2::on(inputs[0])); break;
    case 3: planes = to_vec(gapi::Split3::on(inputs[0])); break;
    case 4: planes = to_vec(gapi::Split4::on(inputs[0])); break;
    default:
        for (int chan = 0; chan < channels; chan++)
            planes.emplace_back(gapi::ChanToPlane::on(inputs[0], chan));
        break;
    }

    return planes;
}

// convert input to interleaved format
std::vector<cv::GMat> merge(const std::vector<cv::GMat>& inputs,
                            int channels) {
    if (inputs.empty()) {
        return inputs;
    }

    std::vector<cv::GMat> interleaved;

    switch (channels) {
    case 1: interleaved.emplace_back(inputs[0]); break;
    case 2: interleaved.emplace_back(gapi::Merge2::on(inputs[0], inputs[1])); break;
    case 3: interleaved.emplace_back(gapi::Merge3::on(inputs[0], inputs[1], inputs[2])); break;
    case 4: interleaved.emplace_back(gapi::Merge4::on(inputs[0], inputs[1], inputs[2], inputs[3])); break;
    default: IE_THROW() << "output channels value " << channels
                                << " is not supported for HWC [by G-API]."
                                << " Expected range (inclusive): [1;4].";
    }

    return interleaved;
}

// validate input/output ColorFormat-related parameters
void validateColorFormats(const G::Desc &in_desc,
                          const G::Desc &out_desc,
                          Layout in_layout,
                          Layout out_layout,
                          ColorFormat input_color_format,
                          ColorFormat output_color_format) {
    const auto verify_desc = [] (const G::Desc& desc, ColorFormat fmt, const std::string& desc_prefix) {
        const auto throw_invalid_number_of_channels = [&](){
            IE_THROW() << desc_prefix << " tensor descriptor "
                               << "has invalid number of channels "
                               << desc.d.C << " for " << fmt
                               << "color format";
        };
        switch (fmt) {
            case ColorFormat::RGB:
            case ColorFormat::BGR: {
                if (desc.d.C != 3) throw_invalid_number_of_channels();
                break;
            }
            case ColorFormat::RGBX:
            case ColorFormat::BGRX: {
                if (desc.d.C != 4) throw_invalid_number_of_channels();
                break;
            }

            default: break;
        }
    };

    const auto verify_layout = [] (Layout layout, const std::string& layout_prefix) {
        if (layout != NHWC && layout != NCHW) {
            IE_THROW() << layout_prefix << " layout " << layout
                               << " is not supported by pre-processing [by G-API]";
        }
    };

    // verify inputs/outputs and throw on error
    const bool color_conv_required = !((output_color_format == input_color_format) || (input_color_format == ColorFormat::RAW));
    if (color_conv_required && (output_color_format == ColorFormat::RAW)) {
        IE_THROW() << "Network's expected color format is unspecified";
    }

    verify_layout(in_layout, "Input blob");
    verify_layout(out_layout, "Network's blob");

    if (!color_conv_required) {
        // verify input and output have the same number of channels
        if (in_desc.d.C != out_desc.d.C) {
            IE_THROW() << "Input and network expected blobs have different number of "
                               << "channels: expected " << out_desc.d.C << " channels but provided "
                               << in_desc.d.C << " channels";
        }
        return;
    }

    // planar 4-channel input is not supported, user can easily pass 3 channels instead of 4
    if (in_layout == NCHW
        && (input_color_format == ColorFormat::RGBX || input_color_format == ColorFormat::BGRX)) {
        IE_THROW() << "Input blob with NCHW layout and BGRX/RGBX color format is "
                           << "explicitly not supported, use NCHW + BGR/RGB color format "
                           << "instead (3 image planes instead of 4)";
    }

    // verify input and output against their corresponding color format
    verify_desc(in_desc, input_color_format, "Input blob");
    verify_desc(out_desc, output_color_format, "Network's blob");
}

bool has_zeros(const SizeVector& vec) {
    return std::any_of(vec.cbegin(), vec.cend(), [] (size_t e) { return e == 0; });
}

void validateTensorDesc(const TensorDesc& desc) {
    auto supports_layout = [](Layout l) { return l == Layout::NCHW || l == Layout::NHWC; };
    const auto layout = desc.getLayout();
    const auto& dims = desc.getDims();
    if (!supports_layout(layout)
        || dims.size() != 4
        || desc.getBlockingDesc().getStrides().size() != 4) {
        IE_THROW() << "Preprocess support NCHW/NHWC only";
    }
    if (has_zeros(dims)) {
        IE_THROW() << "Invalid input data dimensions: "
                           << details::dumpVec(dims);
    }
}

void validateBlob(const MemoryBlob::Ptr &) {}

const std::pair<const TensorDesc&, Layout> getTensorDescAndLayout(const MemoryBlob::Ptr &blob) {
    const auto& desc =  blob->getTensorDesc();
    return {desc, desc.getLayout()};
}

G::Desc getGDesc(G::Desc in_desc_y, const MemoryBlob::Ptr &) {
    return in_desc_y;
}

class PlanarColorConversions {
    using GMats = std::vector<cv::GMat>;
    using CvtFunction = std::function<GMats(const GMats&, Layout, Layout, ResizeAlgorithm)>;
    struct Hash {
        inline size_t operator()(const std::pair<ColorFormat, ColorFormat>& p) const {
            return static_cast<size_t>((p.first << 16) ^ p.second);
        }
    };
    std::unordered_map<std::pair<ColorFormat, ColorFormat>, CvtFunction, Hash> m_conversions;

    // convert RGB -> BGR and BGR -> RGB
    static std::vector<cv::GMat> reverse3(const std::vector<cv::GMat>& inputs,
                                          Layout in_layout,
                                          Layout out_layout,
                                          ResizeAlgorithm algorithm) {
        auto planes = inputs;
        if (in_layout == NHWC) {
            planes = split(inputs, 3);
        }

        // if there's no resize after color convert && output is planar, we have to copy input to
        // output by doing actual G-API operation. otherwise, the graph will be empty (no
        // operations)
        if (algorithm == NO_RESIZE && in_layout == out_layout && out_layout == NCHW) {
            std::vector<cv::GMat> reversed(3);
            reversed[0] = gapi::ChanToPlane::on(planes[2], 0);
            reversed[1] = gapi::ChanToPlane::on(planes[1], 0);
            reversed[2] = gapi::ChanToPlane::on(planes[0], 0);
            return reversed;
        }

        std::reverse(planes.begin(), planes.end());
        return planes;
    }

    // convert RGBX -> RGB and BGRX -> BGR
    static std::vector<cv::GMat> dropLastChan(const std::vector<cv::GMat>& inputs,
                                              Layout in_layout,
                                              Layout out_layout,
                                              ResizeAlgorithm /*algorithm*/) {
        // Note: input is always interleaved, planar input is converted to RGB/BGR on the user side
        auto planes = split(inputs, 4);
        planes.pop_back();
        return planes;
    }

    // convert RGBX -> BGR and BGRX -> RGB
    static std::vector<cv::GMat> dropLastChanAndReverse(const std::vector<cv::GMat>& inputs,
                                                        Layout in_layout,
                                                        Layout out_layout,
                                                        ResizeAlgorithm algorithm) {
        auto planes = dropLastChan(inputs, in_layout, out_layout, algorithm);
        std::reverse(planes.begin(), planes.end());
        return planes;
    }

    static std::vector<cv::GMat> NV12toRGB(const std::vector<cv::GMat>& inputs,
                                           Layout,
                                           Layout,
                                           ResizeAlgorithm) {
        // in_layout is always NCHW
        auto interleaved_rgb = gapi::NV12toRGB::on(inputs[0], inputs[1]);
        return split({interleaved_rgb}, 3);
    }

    static std::vector<cv::GMat> NV12toBGR(const std::vector<cv::GMat>& inputs,
                                           Layout in_layout,
                                           Layout out_layout,
                                           ResizeAlgorithm algorithm) {
        auto planes = NV12toRGB(inputs, in_layout, out_layout, algorithm);
        std::reverse(planes.begin(), planes.end());
        return planes;
    }

    static std::vector<cv::GMat> I420toRGB(const std::vector<cv::GMat>& inputs,
                                           Layout,
                                           Layout,
                                           ResizeAlgorithm) {
        // in_layout is always NCHW
        auto interleaved_rgb = gapi::I420toRGB::on(inputs[0], inputs[1], inputs[2]);
        return split({interleaved_rgb}, 3);
    }

    static std::vector<cv::GMat> I420toBGR(const std::vector<cv::GMat>& inputs,
                                           Layout in_layout,
                                           Layout out_layout,
                                           ResizeAlgorithm algorithm) {
        auto planes = I420toRGB(inputs, in_layout, out_layout, algorithm);
        std::reverse(planes.begin(), planes.end());
        return planes;
    }

public:
    PlanarColorConversions() {
        IE_SUPPRESS_DEPRECATED_START
        m_conversions = {
            { {ColorFormat::RGB, ColorFormat::BGR}, reverse3 },
            { {ColorFormat::BGR, ColorFormat::RGB}, reverse3 },
            { {ColorFormat::RGBX, ColorFormat::RGB}, dropLastChan },
            { {ColorFormat::BGRX, ColorFormat::BGR}, dropLastChan },
            { {ColorFormat::RGBX, ColorFormat::BGR}, dropLastChanAndReverse },
            { {ColorFormat::BGRX, ColorFormat::RGB}, dropLastChanAndReverse }
        };
        IE_SUPPRESS_DEPRECATED_END
    }

    const CvtFunction& at(ColorFormat in_fmt, ColorFormat out_fmt) const {
        auto cvtFunc = m_conversions.find(std::make_pair(in_fmt, out_fmt));
        if (cvtFunc == m_conversions.cend()) {
            IE_THROW() << "Color conversion " << in_fmt << " -> " << out_fmt
                               << " is not supported";
        }
        return cvtFunc->second;
    }
};

// construct G-API graph pipeline to color convert input into output
// Note: always returns planar output
std::vector<cv::GMat>
convertColorPlanar(const std::vector<cv::GMat>& inputs,
                   const G::Desc& in_desc,
                   Layout in_layout,
                   Layout out_layout,
                   ColorFormat input_color_format,
                   ColorFormat output_color_format,
                   ResizeAlgorithm algorithm) {
    // do not perform color conversions if not requested (ColorFormat::RAW) or if input already has
    // network's expected color format
    if (input_color_format == ColorFormat::RAW || input_color_format == output_color_format) {
        // convert input into planar form
        if (in_layout == NHWC) {
            return split(inputs, in_desc.d.C);
        }
        return inputs;
    }

    static PlanarColorConversions conversions;

    const auto& convert = conversions.at(input_color_format, output_color_format);
    auto planes = convert(inputs, in_layout, out_layout, algorithm);
    if (planes.empty()) {
        IE_THROW() << "[G-API] internal error: failed to convert input data into planar "
                           << "format";
    }

    return planes;
}

cv::GComputation buildGraph(const G::Desc &in_desc,
                            const G::Desc &out_desc,
                            Layout in_layout,
                            Layout out_layout,
                            ResizeAlgorithm algorithm,
                            ColorFormat input_color_format,
                            ColorFormat output_color_format) {
    // perform basic validation to ensure our assumptions about input and output are correct
    validateColorFormats(in_desc, out_desc, in_layout, out_layout, input_color_format,
        output_color_format);

    std::vector<cv::GMat> inputs;  // 1 element if NHWC, C elements if NCHW
    if (in_layout == NHWC) {
        inputs.resize(1);
    } else if (in_layout == NCHW) {
        inputs.resize(in_desc.d.C);
    }

    // specific pre-processing case:
    // 1. Requires interleaved image of type CV_8UC3/CV_8UC4 (except for NV12/I420 input)
    // 2. Supports bilinear resize only
    // 3. Supports NV12/I420 -> RGB/BGR color transformations
    const auto io_color_formats = std::make_tuple(input_color_format, output_color_format);
    const bool drop_channel = (io_color_formats == std::make_tuple(ColorFormat::RGBX, ColorFormat::RGB)) ||
                              (io_color_formats == std::make_tuple(ColorFormat::BGRX, ColorFormat::BGR));
    const bool specific_case_of_preproc = ((in_layout == NHWC)
                                        && (in_desc.d.C == 3 || drop_channel)
                                        && ((in_desc.prec == CV_8U) && (in_desc.prec == out_desc.prec))
                                        && (algorithm == RESIZE_BILINEAR)
                                        && (input_color_format == ColorFormat::RAW
                                            || input_color_format == output_color_format
                                            || drop_channel));
    if (specific_case_of_preproc) {
        const auto input_sz = cv::gapi::own::Size(in_desc.d.W, in_desc.d.H);
        const auto scale_sz = cv::gapi::own::Size(out_desc.d.W, out_desc.d.H);

        // convert color format to RGB in case of NV12 input
        std::vector<cv::GMat> color_converted_input;
        color_converted_input = inputs;

        auto planes = drop_channel ?
                to_vec(gapi::ScalePlanes4:: on(
                        color_converted_input[0], in_desc.prec, input_sz, scale_sz, cv::INTER_LINEAR))
              : to_vec(gapi::ScalePlanes  ::on(
                        color_converted_input[0], in_desc.prec, input_sz, scale_sz, cv::INTER_LINEAR));

        if (drop_channel) {
            planes.pop_back();
        }

        std::vector<cv::GMat> outputs;
        if (out_layout == NHWC) {
            outputs.emplace_back(gapi::Merge3::on(planes[0], planes[1], planes[2]));
        } else {
            outputs = planes;
        }
        return cv::GComputation(inputs, outputs);
    }

    auto planes = convertColorPlanar(inputs, in_desc, in_layout, out_layout,
        input_color_format, output_color_format, algorithm);

    const int number_of_planes = static_cast<int>(planes.size());
    if (number_of_planes != out_desc.d.C) {
        IE_THROW() << "[G-API] internal error: number of channels after color conversion "
                           << "!= network's expected number of channels: "
                           << number_of_planes << " != " << out_desc.d.C;
    }

    const int tmp_prec = CV_32F;

    std::vector<cv::GMat> outputs;
    const bool resize_needed = (algorithm != NO_RESIZE);
    const bool need_tmp_prec_conv = resize_needed && (in_desc.prec != CV_8U) && (in_desc.prec != CV_32F);

    if (resize_needed) {
        // resize every plane
        std::vector<cv::GMat> out_planes;
        out_planes.reserve(planes.size());
        const int interp_type = [](const ResizeAlgorithm &ar) {
            switch (ar) {
            case RESIZE_AREA:     return cv::INTER_AREA;
            case RESIZE_BILINEAR: return cv::INTER_LINEAR;
            default: IE_THROW() << "Unsupported resize operation";
            }
        } (algorithm);

        std::transform(planes.begin(), planes.end(), std::back_inserter(out_planes), [&](const cv::GMat& m) {
            const auto input_sz  = cv::gapi::own::Size(in_desc.d.W, in_desc.d.H);
            const auto scale_sz  = cv::gapi::own::Size(out_desc.d.W, out_desc.d.H);

            cv::GMat converted = m;
            int prec = in_desc.prec;

            if (need_tmp_prec_conv) {
                std::tie(converted, prec) = std::make_tuple(gapi::ConvertDepth::on(m, tmp_prec), tmp_prec);
            }

            return gapi::ScalePlane::on(converted, prec, input_sz, scale_sz, interp_type);
        });
        outputs = out_planes;
    } else {
        outputs = planes;
    }

    if ((in_desc.prec != out_desc.prec) || need_tmp_prec_conv) {
        auto convert_prec = [](const std::vector<cv::GMat> & src_gmats, int dst_precision) {
            std::vector<cv::GMat> dst_gmats;
            std::transform(src_gmats.begin(), src_gmats.end(), std::back_inserter(dst_gmats), [&](cv::GMat const& m){
                return gapi::ConvertDepth::on(m, dst_precision);
            });
            return dst_gmats;
        };

        outputs = convert_prec(outputs, out_desc.prec);
    }
    // convert to interleaved if NHWC is required as output
    if (out_layout == NHWC) {
        outputs = merge(outputs, out_desc.d.C);
    }

    return cv::GComputation(inputs, outputs);
}
}  // anonymous namespace

PreprocEngine::PreprocEngine() : _lastComp(parallel_get_max_threads()) {}

PreprocEngine::Update PreprocEngine::needUpdate(const CallDesc &newCallOrig) const {
    // Given our knowledge about Fluid, full graph rebuild is required
    // if and only if:
    // 0. This is the first call ever
    // 1. precision has changed (affects kernel versions)
    // 2. layout has changed (affects graph topology)
    // 3. algorithm has changed (affects kernel version)
    // 4. dimensions have changed from downscale to upscale or vice-versa if interpolation is AREA
    // 5. color format has changed (affects graph topology)
    if (!_lastCall) {
        return Update::REBUILD;
    }

    BlobDesc last_in;
    BlobDesc last_out;
    ResizeAlgorithm last_algo = ResizeAlgorithm::NO_RESIZE;
    std::tie(last_in, last_out, last_algo) = *_lastCall;

    CallDesc newCall = newCallOrig;
    BlobDesc new_in;
    BlobDesc new_out;
    ResizeAlgorithm new_algo = ResizeAlgorithm::NO_RESIZE;
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

void PreprocEngine::checkApplicabilityGAPI(const Blob::Ptr &src, const Blob::Ptr &dst) {
    // Note: src blob is the ROI blob, dst blob is the network's input blob

    if (!src->is<MemoryBlob>()) {
        IE_THROW()  << "Unsupported input blob type: expected MemoryBlob";
    }

    // dst is always a memory blob
    if (!dst->is<MemoryBlob>()) {
        IE_THROW()  << "Unsupported network's input blob type: expected MemoryBlob";
    }

    const auto &src_dims = src->getTensorDesc().getDims();
    const auto &dst_dims = dst->getTensorDesc().getDims();

    // dimensions sizes must be equal if both blobs are memory blobs
    if (src_dims.size() != dst_dims.size()) {
        IE_THROW() << "Preprocessing is not applicable. Source and destination blobs "
                              "have different number of dimensions.";
    }
    if (dst_dims.size() != 4) {
        IE_THROW() << "Preprocessing is not applicable. Only 4D tensors are supported.";
    }

    // dimensions must not have values that are equal to 0
    if (has_zeros(src_dims)) {
        IE_THROW() << "Invalid input data dimensions: " << details::dumpVec(src_dims);
    }

    if (has_zeros(dst_dims)) {
        IE_THROW() << "Invalid network's input dimensions: " << details::dumpVec(dst_dims);
    }
}

int PreprocEngine::getCorrectBatchSize(int batch, const Blob::Ptr& blob) {
    if (batch == 0) {
        IE_THROW() << "Input pre-processing is called with invalid batch size " << batch;
    }

    if (blob->is<CompoundBlob>()) {
        // batch size must always be 1 in compound blob case
        if (batch > 1) {
            IE_THROW()  << "Provided input blob batch size " << batch
                                << " is not supported in compound blob pre-processing";
        }
        batch = 1;
    } else if (batch < 0) {
        // if batch size is unspecified, process the whole input blob
        batch = static_cast<int>(blob->getTensorDesc().getDims()[0]);
    }

    return batch;
}

void PreprocEngine::executeGraph(Opt<cv::GComputation>& lastComputation,
    const std::vector<std::vector<cv::gapi::own::Mat>>& batched_input_plane_mats,
    std::vector<std::vector<cv::gapi::own::Mat>>& batched_output_plane_mats, int batch_size, bool omp_serial,
    Update update) {

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
    parallel_nt_static(thread_num, [&, this](int slice_n, const int total_slices) {
        OV_ITT_SCOPED_TASK(itt::domains::IEPreproc, _perf_exec_tile);

        auto& compiled = _lastComp[slice_n];
        if (Update::REBUILD == update || Update::RESHAPE == update) {
            //  need to compile (or reshape) own object for a particular ROI
            OV_ITT_SCOPED_TASK(itt::domains::IEPreproc, _perf_graph_compiling);

            using cv::gapi::own::Rect;

            // current design implies all images in batch are equal
            const auto& input_plane_mats = batched_input_plane_mats[0];
            const auto& output_plane_mats = batched_output_plane_mats[0];

            auto lines_per_thread = output_plane_mats[0].rows / total_slices;
            const auto remainder = output_plane_mats[0].rows % total_slices;

            // remainder shows how many threads must calculate 1 additional row. now these additions
            // must also be addressed in rect's Y coordinate:
            int roi_y = 0;
            if (slice_n < remainder) {
                lines_per_thread++;  // 1 additional row
                roi_y = slice_n * lines_per_thread;  // all previous rois have lines+1 rows
            } else {
                // remainder rois have lines+1 rows, the rest prior to slice_n have lines rows
                roi_y =
                    remainder * (lines_per_thread + 1) + (slice_n - remainder) * lines_per_thread;
            }

            if (lines_per_thread <= 0) return;  // no job for current thread

            auto roi = Rect{0, roi_y, output_plane_mats[0].cols, lines_per_thread};
            std::vector<Rect> rois(output_plane_mats.size(), roi);

            // TODO: make a ROI a runtime argument to avoid
            // recompilations
            auto args = cv::compile_args(gapi::preprocKernels(), cv::GFluidOutputRois{std::move(rois)});
            if (Update::REBUILD == update) {
                auto& computation = lastComputation.value();
                compiled = computation.compile(descrs_of(input_plane_mats), std::move(args));
            } else {
                IE_ASSERT(compiled);
                compiled.reshape(descrs_of(input_plane_mats), std::move(args));
            }
        }

        for (int i = 0; i < batch_size; ++i) {
            const auto& input_plane_mats = batched_input_plane_mats[i];
            auto& output_plane_mats = batched_output_plane_mats[i];

            cv::GRunArgs call_ins;
            cv::GRunArgsP call_outs;
            for (const auto & m : input_plane_mats) { call_ins.emplace_back(m);}
            for (auto & m : output_plane_mats) { call_outs.emplace_back(&m);}

            OV_ITT_SCOPED_TASK(itt::domains::IEPreproc, _perf_exec_graph);
            compiled(std::move(call_ins), std::move(call_outs));
        }
    });
}

template<typename BlobTypePtr>
void PreprocEngine::preprocessBlob(const BlobTypePtr &inBlob, MemoryBlob::Ptr &outBlob,
    ResizeAlgorithm algorithm, ColorFormat in_fmt, ColorFormat out_fmt, bool omp_serial,
    int batch_size) {

    validateBlob(inBlob);

    auto desc_and_layout = getTensorDescAndLayout(inBlob);

    const auto& in_desc_ie = desc_and_layout.first;
    const auto  in_layout  = desc_and_layout.second;

    const auto& out_desc_ie = outBlob->getTensorDesc();
    validateTensorDesc(in_desc_ie);
    validateTensorDesc(out_desc_ie);


    const auto out_layout = out_desc_ie.getLayout();

    // For YUV420, check batch via Y plane descriptor
    const G::Desc
        in_desc =  G::decompose(in_desc_ie),
        out_desc = G::decompose(out_desc_ie);

    // according to the IE's current design, input blob batch size _must_ match networks's expected
    // batch size, even if the actual processing batch size (set on infer request) is different.
    if (in_desc.d.N != out_desc.d.N) {
        IE_THROW()  << "Input blob batch size is invalid: (input blob) "
                            << in_desc.d.N << " != " << out_desc.d.N << " (expected by network)";
    }

    // sanity check batch size
    if (batch_size > out_desc.d.N) {
        IE_THROW()  << "Provided batch size is invalid: (provided)"
                            << batch_size << " > " << out_desc.d.N << " (expected by network)";
    }

    CallDesc thisCall = CallDesc{ BlobDesc{ in_desc_ie.getPrecision(),
                                            in_layout,
                                            in_desc_ie.getDims(),
                                            in_fmt },
                                  BlobDesc{ out_desc_ie.getPrecision(),
                                            out_layout,
                                            out_desc_ie.getDims(),
                                            out_fmt },
                                  algorithm };

    if (algorithm == NO_RESIZE && std::get<0>(thisCall) == std::get<1>(thisCall)) {
        //if requested output parameters match input blob no need to do anything
        IE_THROW()  << "No job to do in the PreProcessing ?";
    }

    const Update update = needUpdate(thisCall);

    Opt<cv::GComputation> _lastComputation;
    if (Update::REBUILD == update || Update::RESHAPE == update) {
        _lastCall = cv::util::make_optional(std::move(thisCall));

        if (Update::REBUILD == update) {
            //  rebuild the graph
            OV_ITT_SCOPED_TASK(itt::domains::IEPreproc, _perf_graph_building);
            // FIXME: what is a correct G::Desc to be passed for NV12/I420 case?
            auto custom_desc = getGDesc(in_desc, inBlob);
            _lastComputation = cv::util::make_optional(
                buildGraph(custom_desc,
                           out_desc,
                           in_layout,
                           out_layout,
                           algorithm,
                           in_fmt,
                           out_fmt));
        }
    }

    auto batched_input_plane_mats  = bind_to_blob(inBlob,  batch_size);
    auto batched_output_plane_mats = bind_to_blob(outBlob, batch_size);

    executeGraph(_lastComputation, batched_input_plane_mats, batched_output_plane_mats, batch_size,
        omp_serial, update);
}

void PreprocEngine::preprocessWithGAPI(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
        const ResizeAlgorithm& algorithm, ColorFormat in_fmt, bool omp_serial, int batch_size) {
    const auto out_fmt = (in_fmt == ColorFormat::RAW) ? ColorFormat::RAW : ColorFormat::BGR;  // FIXME: get expected color format from network

    // output is always a memory blob
    auto outMemoryBlob = as<MemoryBlob>(outBlob);
    if (!outMemoryBlob) {
        IE_THROW()  << "Unsupported network's input blob type: expected MemoryBlob";
    }

    auto inMemoryBlob = as<MemoryBlob>(inBlob);
    if (!inMemoryBlob) {
        IE_THROW()  << "Unsupported input blob for color format " << in_fmt
            << ": expected MemoryBlob";
    }
    return preprocessBlob(inMemoryBlob, outMemoryBlob, algorithm, in_fmt, out_fmt, omp_serial,
                          batch_size);
}
}  // namespace InferenceEngine
