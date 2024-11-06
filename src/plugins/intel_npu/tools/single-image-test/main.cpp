//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "image_quality_helper.hpp"
#include "openvino/core/partial_shape.hpp"
#include "semantic_segmentation_helpers.hpp"
#include "tensor_utils.hpp"
#include "yolo_helpers.hpp"
#include "tools_helpers.hpp"

#include <openvino/core/parallel.hpp>
#include <openvino/openvino.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <gflags/gflags.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using TensorMap = std::map<std::string, ov::Tensor>;

struct TensorDescriptor {
    ov::element::Type precision;
    ov::PartialShape shape;
    ov::Shape dataShape;
    ov::Layout layout;
};

using TensorDescriptorMap = std::unordered_map<std::string, TensorDescriptor>;
using LayoutMap = std::unordered_map<std::string, ov::Layout>;

/**
 * @brief Provides a caseless equality function for STL algorithms.
 * @details Utility function taken from the OpenVINO implementation, formerly registered as
 * "InferenceEngine::details::CaselessEq"
 * @tparam Key
 */
template <class Key>
class CaselessEq {
public:
    bool operator()(const Key& a, const Key& b) const noexcept {
        return a.size() == b.size() &&
               std::equal(std::begin(a), std::end(a), std::begin(b), [](const char cha, const char chb) {
                   return std::tolower(cha) == std::tolower(chb);
               });
    }
};

CaselessEq<std::string> strEq;

//
// Command line options
//

DEFINE_string(network, "", "Network file (either XML or pre-compiled blob)");
DEFINE_string(input, "", "Input file(s)");
DEFINE_string(compiled_blob, "", "Output compiled network file (compiled result blob)");
DEFINE_uint32(override_model_batch_size, 1, "Enforce a model to be compiled for batch size");
DEFINE_string(device, "", "Device to use");
DEFINE_string(config, "", "Path to the configuration file (optional)");
DEFINE_string(ip, "", "Input precision (default: U8, available: FP32, FP16, I32, I64, U8)");
DEFINE_string(op, "", "Output precision (default: FP32, available: FP32, FP16, I32, I64, U8)");
DEFINE_string(
        il, "",
        "Input layout for all inputs, or ';' separated list of pairs <input>:<layout>. Regex in <input> is supported");
DEFINE_string(ol, "",
              "Output layout for all outputs, or ';' separated list of pairs <output>:<layout>. Regex in <output> is "
              "supported");
DEFINE_string(iml, "",
              "Model input layout for all model inputs, or ';' separated list of pairs <input>:<layout>. Regex in "
              "<input> is supported");
DEFINE_string(oml, "",
              "Model output layout for all outputs, or ';' separated list of pairs <output>:<layout>. Regex in "
              "<output> is supported");
DEFINE_bool(img_as_bin, false, "Force binary input even if network expects an image");
DEFINE_bool(pc, false, "Report performance counters");
DEFINE_string(
        shape, "",
        "Optional. Set shape for model input. For example, \"input1[1,3,224,224],input2[1,4]\" or \"[1,3,224,224]\""
        " in case of one input size. This parameter affects model input shape and can be dynamic."
        " For dynamic dimensions use symbol `?` or '-1'. Ex. [?,3,?,?]."
        " For bounded dimensions specify range 'min..max'. Ex. [1..10,3,?,?].");
DEFINE_string(data_shape, "",
    "Required for models with dynamic shapes. Set shape for input blobs. Only one shape can be set."
    "In case of one input size: \"[1,3,224,224]\"");

// for using input image mean and scale
static constexpr char mean_values_message[] =
        "Optional. Mean values to be used for the input image per channel. "
        "Values to be provided in the [channel1,channel2,channel3] format. "
        "Can be defined for desired input of the model, for example: \"--mean_values "
        "data[255,255,255],info[255,255,255]\". The exact meaning and order of channels depend on how the original "
        "model "
        "was trained. Applying the values affects performance and may cause type conversion";

static constexpr char scale_values_message[] =
        "Optional. Scale values to be used for the input image per channel. "
        "Values are provided in the [channel1,channel2,channel3] format. "
        "Can be defined for desired input of the model, for example: \"--scale_values "
        "data[255,255,255],info[255,255,255]\". "
        "The exact meaning and order of channels depend on how the original model was trained. If both --mean_values "
        "and "
        "--scale_values are specified, the mean is subtracted first and then scale is applied regardless of the order "
        "of "
        "options in command line. Applying the values affects performance and may cause type conversion";
DEFINE_string(mean_values, "", mean_values_message);
DEFINE_string(scale_values, "", scale_values_message);

DEFINE_string(img_bin_precision, "", "Specify the precision of the binary input files. Eg: 'FP32,FP16,I32,I64,U8'");

DEFINE_bool(run_test, false, "Run the test (compare current results with previously dumped)");
DEFINE_string(
    ref_dir,
    "",
    "A directory with reference blobs to compare with in run_test mode. Leave it empty to use the current folder.");
DEFINE_string(mode, "", "Comparison mode to use");

DEFINE_uint32(top_k, 1, "Top K parameter for 'classification' mode");
DEFINE_double(prob_tolerance, 1e-4, "Probability tolerance for 'classification/ssd/yolo' mode");

DEFINE_double(raw_tolerance, 1e-4, "Tolerance for 'raw' mode (absolute diff)");
DEFINE_double(cosim_threshold, 0.90, "Threshold for 'cosim' mode");
DEFINE_double(rrmse_loss_threshold, std::numeric_limits<double>::max(), "Threshold for 'rrmse' mode");
DEFINE_double(nrmse_loss_threshold, 1.0, "Threshold for 'nrmse' mode");
DEFINE_double(confidence_threshold, 1e-4, "Confidence threshold for Detection mode");
DEFINE_double(box_tolerance, 1e-4, "Box tolerance for 'detection' mode");

DEFINE_double(psnr_reference, 30.0, "PSNR reference value in dB");
DEFINE_double(psnr_tolerance, 1e-4, "Tolerance for 'psnr' mode");

DEFINE_string(log_level, "", "IE logger level (optional)");
DEFINE_string(color_format, "BGR", "Color format for input: RGB or BGR");
DEFINE_uint32(scale_border, 4, "Scale border");
DEFINE_bool(normalized_image, false, "Images in [0, 1] range or not");

// for Yolo
DEFINE_bool(is_tiny_yolo, false, "Is it Tiny Yolo or not (true or false)?");
DEFINE_int32(classes, 80, "Number of classes for Yolo V3");
DEFINE_int32(coords, 4, "Number of coordinates for Yolo V3");
DEFINE_int32(num, 3, "Number of scales for Yolo V3");

typedef std::chrono::high_resolution_clock Time;
// for Semantic Segmentation
DEFINE_uint32(sem_seg_classes, 12, "Number of classes for semantic segmentation");
DEFINE_double(sem_seg_threshold, 0.98, "Threshold for 'semantic segmentation' mode");
DEFINE_uint32(sem_seg_ignore_label, std::numeric_limits<uint32_t>::max(), "The number of the label to be ignored");
DEFINE_string(dataset, "NONE",
              "The dataset used to train the model. Useful for instances such as semantic segmentation to visualize "
              "the accuracy per-class");
std::vector<std::string> camVid12 = {"Sky",        "Building", "Pole", "Road",       "Pavement",  "Tree",
                                     "SignSymbol", "Fence",    "Car",  "Pedestrian", "Bicyclist", "Unlabeled"};

std::vector<std::string> splitStringList(const std::string& str, char delim) {
    std::vector<std::string> out;

    if (str.empty()) {
        return out;
    }

    std::istringstream istr(str);

    std::string elem;
    while (std::getline(istr, elem, delim)) {
        if (elem.empty()) {
            continue;
        }

        out.push_back(std::move(elem));
    }

    return out;
}

std::map<std::string, std::string> parseArgMap(std::string argMap) {
    argMap.erase(std::remove_if(argMap.begin(), argMap.end(), ::isspace), argMap.end());

    const auto pairs = splitStringList(argMap, ';');

    std::map<std::string, std::string> parsedMap;
    for (auto&& pair : pairs) {
        const auto lastDelimPos = pair.find_last_of(':');
        auto key = pair.substr(0, lastDelimPos);
        std::string value;
        if (lastDelimPos != std::string::npos) {
            value = pair.substr(lastDelimPos + 1);
        }
        parsedMap[std::move(key)] = std::move(value);
    }

    return parsedMap;
}

void parseCommandLine(int argc, char* argv[]) {
    std::ostringstream usage;
    usage << "Usage: " << argv[0] << "[<options>]";
    gflags::SetUsageMessage(usage.str());

    std::ostringstream version;
    version << ov::get_openvino_version();
    gflags::SetVersionString(version.str());

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << "Parameters:" << std::endl;
    std::cout << "    Network file:                             " << FLAGS_network << std::endl;
    std::cout << "    Input file(s):                            " << FLAGS_input << std::endl;
    std::cout << "    Output compiled network file:             " << FLAGS_compiled_blob << std::endl;
    std::cout << "    Color format:                             " << FLAGS_color_format << std::endl;
    std::cout << "    Input precision:                          " << FLAGS_ip << std::endl;
    std::cout << "    Output precision:                         " << FLAGS_op << std::endl;
    std::cout << "    Input layout:                             " << FLAGS_il << std::endl;
    std::cout << "    Output layout:                            " << FLAGS_ol << std::endl;
    std::cout << "    Model input layout:                       " << FLAGS_iml << std::endl;
    std::cout << "    Model output layout:                      " << FLAGS_oml << std::endl;
    std::cout << "    Img as binary:                            " << FLAGS_img_as_bin << std::endl;
    std::cout << "    Bin input file precision:                 " << FLAGS_img_bin_precision << std::endl;
    std::cout << "    Device:                                   " << FLAGS_device << std::endl;
    std::cout << "    Config file:                              " << FLAGS_config << std::endl;
    std::cout << "    Run test:                                 " << FLAGS_run_test << std::endl;
    std::cout << "    Performance counters:                     " << FLAGS_pc << std::endl;
    std::cout << "    Mean_values [channel1,channel2,channel3]  " << FLAGS_mean_values << std::endl;
    std::cout << "    Scale_values [channel1,channel2,channel3] " << FLAGS_scale_values << std::endl;
    if (FLAGS_run_test) {
        std::cout << "    Reference files direcotry:                "
                  << (FLAGS_ref_dir.empty() ? "Current directory" : FLAGS_ref_dir) << std::endl;
        std::cout << "    Mode:             " << FLAGS_mode << std::endl;
        if (strEq(FLAGS_mode, "classification")) {
            std::cout << "    Top K:            " << FLAGS_top_k << std::endl;
            std::cout << "    Tolerance:        " << FLAGS_prob_tolerance << std::endl;
        } else if (strEq(FLAGS_mode, "raw")) {
            std::cout << "    Tolerance:        " << FLAGS_raw_tolerance << std::endl;
        } else if (strEq(FLAGS_mode, "cosim")) {
            std::cout << "    Threshold:        " << FLAGS_cosim_threshold << std::endl;
        } else if (strEq(FLAGS_mode, "psnr")) {
            std::cout << "    Reference:        " << FLAGS_psnr_reference << std::endl;
            std::cout << "    Tolerance:        " << FLAGS_psnr_tolerance << std::endl;
            std::cout << "    Scale_border:     " << FLAGS_scale_border << std::endl;
            std::cout << "    Normalized_image: " << FLAGS_normalized_image << std::endl;
        } else if (strEq(FLAGS_mode, "rrmse")) {
            std::cout << "    Threshold:        " << FLAGS_rrmse_loss_threshold << std::endl;
        } else if (strEq(FLAGS_mode, "nrmse")) {
            std::cout << "    Threshold:        " << FLAGS_nrmse_loss_threshold << std::endl;
        }
    }
    std::cout << "    Log level:                        " << FLAGS_log_level << std::endl;
    std::cout << std::endl;
}

//
// OpenCV to OpenVINO conversion
//

bool isImage(const ov::Shape& shape, const ov::Layout& layout) {
    if (shape.size() == 4) {
        const auto numChannels = shape[ov::layout::channels_idx(layout)];
        return (numChannels == 3) || (numChannels == 4);
    }

    return false;
}

/**
 * @brief Computes the offsets for each axis in terms of number of elements.
 * @details Taken from the OpenVINO implementation ("InferenceEngine::BlockingDesc::fillDesc") and slightly modified for
 * the current usecase.
 *
 * @param shape The shape based on which the offsets will be computed.
 * @return The resulting strides.
 */
std::vector<size_t> getStrides(const ov::Shape& shape) {
    std::vector<size_t> strides(shape.size());

    strides[strides.size() - 1] = 1;

    for (size_t i = 2; i <= shape.size(); i++) {
        strides[strides.size() - i] = strides[strides.size() - (i - 1)] * shape[shape.size() - (i - 1)];
    }

    return strides;
}

std::vector<cv::Mat> ovToCV(const ov::Tensor& tensor, const ov::Shape& shape, const ov::Layout& layout,
                            size_t batchInd = 0, size_t depthInd = 0) {
    const ov::element::Type& precision = tensor.get_element_type();

    OPENVINO_ASSERT(layout == ov::Layout("NCHW") || layout == ov::Layout("NCDHW"),
                    "Unsupported layout: ", layout.to_string());

    OPENVINO_ASSERT(precision == ov::element::Type_t::u8 || precision == ov::element::Type_t::f32 ||
                            precision == ov::element::Type_t::f16 || precision == ov::element::Type_t::bf16 ||
                            precision == ov::element::Type_t::i32,
                    "Unsupported precision: ", precision.get_type_name());

    int cvType = 0;
    size_t elemSize = 0;

    if (precision == ov::element::Type_t::u8) {
        cvType = CV_8UC1;
        elemSize = sizeof(uint8_t);
    } else if (precision == ov::element::Type_t::f32) {
        cvType = CV_32FC1;
        elemSize = sizeof(float);
    } else if (precision == ov::element::Type_t::f16) {
        cvType = CV_16SC1;
        elemSize = sizeof(ov::float16);
    } else if (precision == ov::element::Type_t::bf16) {
        cvType = CV_16SC1;
        elemSize = sizeof(ov::bfloat16);
    } else if (precision == ov::element::Type_t::i32) {
        cvType = CV_32SC1;
        elemSize = sizeof(int32_t);
    }

    std::vector<cv::Mat> out;

    const size_t N = shape[ov::layout::batch_idx(layout)];
    const size_t C = shape[ov::layout::channels_idx(layout)];
    const size_t H = shape[ov::layout::height_idx(layout)];
    const size_t W = shape[ov::layout::width_idx(layout)];

    const auto dataBuffer = reinterpret_cast<uint8_t*>(tensor.data());

    if (layout == ov::Layout("NCHW")) {
        OPENVINO_ASSERT(batchInd < N);
        OPENVINO_ASSERT(C == 3 || C == 4, "Unsupported number of channels: ", C);

        out.resize(C);
        for (size_t c = 0; c < C; ++c) {
            out[c] = cv::Mat(static_cast<int>(H), static_cast<int>(W), cvType,
                             dataBuffer + (batchInd * C + c) * W * H * elemSize);
        }
    } else if (layout == ov::Layout("NCDHW")) {
        const size_t D = shape[ov::layout::depth_idx(layout)];

        const std::vector<size_t> strides = getStrides(shape);

        const auto strideN = strides[ov::layout::batch_idx(layout)];
        const auto strideC = strides[ov::layout::channels_idx(layout)];
        const auto strideD = strides[ov::layout::depth_idx(layout)];

        OPENVINO_ASSERT(batchInd < N);
        OPENVINO_ASSERT(depthInd < D);
        OPENVINO_ASSERT(C == 3 || C == 4, "Unsupported number of channels: ", C);

        out.resize(C);
        for (size_t c = 0; c < C; ++c) {
            out[c] = cv::Mat(static_cast<int>(H), static_cast<int>(W), cvType,
                             dataBuffer + (strideN * batchInd + strideC * c + strideD * depthInd) * elemSize);
        }
    }

    return out;
}

/**
 * @brief Converts the source data from its current precision to the one used by the destination buffer and then places
 * the result inside it.
 * @details The conversion is performed by the use of the "static_cast" operator. Depending on the types between which
 * the conversions are made, this may lead to undefined behavior.
 *
 * E.g.: Several experiments suggest float<->ov::float16 conversions may work fine, but float->uint8_t may lead to
 * division by zero.
 *
 * @tparam InT The type of the source buffer
 * @tparam OutT The type of the destination buffer
 * @param destination Where the result will be stored
 * @param source The data which shall be converted
 * @param numberOfElements Indicates how many elements will be taken from the source buffer.
 */
template <typename InT, typename OutT>
void convertBufferType(OutT* destination, const InT* source, size_t numberOfElements) {
    ov::parallel_for(numberOfElements, [source, destination](int64_t index) {
        destination[index] = static_cast<OutT>(source[index]);
    });
}

void cvToOV(const cv::Mat& cvImg, const ov::Tensor& tensor, const ov::Shape& shape, const ov::Layout& layout,
            const std::string& colorFormat) {
    const ov::element::Type& precision = tensor.get_element_type();

    OPENVINO_ASSERT(layout == ov::Layout("NHWC") || layout == ov::Layout("NCHW"),
                    "Unsupported layout: ", layout.to_string());

    const auto N = shape[ov::layout::batch_idx(layout)];
    const auto C = shape[ov::layout::channels_idx(layout)];
    const auto H = shape[ov::layout::height_idx(layout)];
    const auto W = shape[ov::layout::width_idx(layout)];

    OPENVINO_ASSERT(C == 3 || C == 4, "Unsupported number of channels: ", C);

    int cvType = 0;

    if (precision == ov::element::Type_t::u8) {
        cvType = static_cast<int>(CV_8UC(C));
    } else if (precision == ov::element::Type_t::f32) {
        cvType = static_cast<int>(CV_32FC(C));
    } else if (precision == ov::element::Type_t::f16) {
        cvType = static_cast<int>(CV_16SC(C));
    } else if (precision == ov::element::Type_t::bf16) {
        cvType = static_cast<int>(CV_16SC(C));
    } else if (precision == ov::element::Type_t::i32) {
        cvType = static_cast<int>(CV_32SC(C));
    } else {
        OPENVINO_ASSERT(precision == ov::element::Type_t::u8 || precision == ov::element::Type_t::f32 ||
                                precision == ov::element::Type_t::f16 || precision == ov::element::Type_t::bf16 ||
                                precision == ov::element::Type_t::i32,
                        "Unsupported precision ", precision.get_type_name());
    }

    cv::Mat in;

    if (C == 3) {
        if (colorFormat == "RGB") {
            cv::cvtColor(cvImg, in, cv::COLOR_BGR2RGB);
        } else {
            in = cvImg;
        }
    } else {
        if (colorFormat == "RGB") {
            cv::cvtColor(cvImg, in, cv::COLOR_BGR2RGBA);
        } else {
            cv::cvtColor(cvImg, in, cv::COLOR_BGR2BGRA);
        }
    }

    if (precision != ov::element::Type_t::u8) {
        in.convertTo(in, CV_32F);
    }

    const auto pictureArea = static_cast<size_t>(in.size().area());

    if (W * H > pictureArea) {
        cv::resize(in, in, cv::Size(static_cast<int>(W), static_cast<int>(H)), 0.0, 0.0, cv::INTER_AREA);
    } else {
        cv::resize(in, in, cv::Size(static_cast<int>(W), static_cast<int>(H)), 0.0, 0.0, cv::INTER_LINEAR);
    }

    if (layout == ov::Layout("NHWC")) {
        const auto dataBuffer = reinterpret_cast<uint8_t*>(tensor.data());

        cv::Mat out(static_cast<int>(H), static_cast<int>(W), cvType, dataBuffer);

        if (precision == ov::element::Type_t::f16) {
            const auto inPtr = in.ptr<float>();
            const auto outPtr = out.ptr<ov::float16>();
            convertBufferType(outPtr, inPtr, out.size().area() * C);
        } else if (precision == ov::element::Type_t::bf16) {
            const auto inPtr = in.ptr<float>();
            const auto outPtr = out.ptr<ov::bfloat16>();
            convertBufferType(outPtr, inPtr, out.size().area() * C);
        } else if (precision == ov::element::Type_t::i32) {
            in.convertTo(out, CV_32S);
        } else {
            in.copyTo(out);
        }

        for (size_t n = 1; n < N; ++n) {
            cv::Mat batch(static_cast<int>(H), static_cast<int>(W), cvType,
                          dataBuffer + n * (out.size().area() * out.elemSize()));
            out.copyTo(batch);
        }
    } else if (layout == ov::Layout("NCHW")) {
        auto tensorPlanes = ovToCV(tensor, shape, layout, 0);

        if (!(precision == ov::element::Type_t::f16 ||
            precision == ov::element::Type_t::bf16)) {
            cv::split(in, tensorPlanes);
        } else {
            std::vector<cv::Mat> inPlanes;
            cv::split(in, inPlanes);

            OPENVINO_ASSERT(tensorPlanes.size() == inPlanes.size());

            for (size_t i = 0; i < tensorPlanes.size(); ++i) {
                const auto inPtr = inPlanes[i].ptr<float>();
                if (precision == ov::element::Type_t::f16) {
                    const auto outPtr = tensorPlanes[i].ptr<ov::float16>();
                    convertBufferType(outPtr, inPtr, inPlanes[i].size().area());
                } else if (precision == ov::element::Type_t::bf16) {
                    const auto outPtr = tensorPlanes[i].ptr<ov::bfloat16>();
                    convertBufferType(outPtr, inPtr, inPlanes[i].size().area());
                }
            }
        }

        for (size_t n = 1; n < N; ++n) {
            const auto batchPlanes = ovToCV(tensor, shape, layout, n);

            OPENVINO_ASSERT(batchPlanes.size() == tensorPlanes.size());

            for (size_t i = 0; i < tensorPlanes.size(); ++i) {
                tensorPlanes[i].copyTo(batchPlanes[i]);
            }
        }
    }
}

std::vector<float> splitFloat(const std::string& s, char delim) {
    std::vector<float> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(std::stof(item));
    }
    return result;
}

std::unordered_map<std::string, std::vector<float>> parseMeanOrScaleString(const std::string& mean_scale) {
    std::unordered_map<std::string, std::vector<float>> result;

    //  Format: layer1[255,255,255],layer2[255,255,255] for particular layers,
    //          or [255,255,255] for all layers
    std::string search_string = mean_scale;
    auto start_pos = search_string.find_first_of('[');
    while (start_pos != std::string::npos) {
        auto end_pos = search_string.find_first_of(']');
        if (end_pos == std::string::npos)
            break;
        auto input_name = search_string.substr(0, start_pos);
        if (result.count(input_name) == 0) {
            auto input_value_string = search_string.substr(start_pos + 1, end_pos - start_pos - 1);
            result[input_name] = splitFloat(input_value_string, ',');
            if (input_name.empty()) {
                if (mean_scale != search_string) {
                    throw std::logic_error("Can't parse input parameter string: " + mean_scale +
                                           ". Format of value: layer1[255,255,255],layer2[255,255,255] "
                                           "for particular layers, or just [255,255,255] for all layers.");
                }
                search_string = search_string.substr(end_pos + 1);
                break;
            }
        } else {
            throw std::logic_error("Specifying mean and scale for the same layer/s"
                                   " more than once is prohibited: " +
                                   mean_scale);
        }

        search_string = search_string.substr(end_pos + 1);
        if (search_string.empty() || search_string.front() != ',')
            break;
        search_string = search_string.substr(1);
        if (search_string.empty()) {
            throw std::logic_error("Can't parse input parameter string: " + mean_scale +
                                   ". Format of value: layer1[255,255,255],layer2[255,255,255] "
                                   "for particular layers, or just [255,255,255] for all layers.");
        }
        start_pos = search_string.find_first_of('[');
    }
    if (!search_string.empty())
        throw std::logic_error("Can't parse input parameter string: " + mean_scale +
                               ". Format of value: layer1[255,255,255],layer2[255,255,255] "
                               "for particular layers, or just [255,255,255] for all layers.");

    return result;
}

std::vector<std::vector<float>> parseMeanOrScale(const std::string& mean_scale,
                                                 const std::vector<ov::Output<ov::Node>>& inputs_info) {
    std::vector<std::vector<float>> result(inputs_info.size());

    auto mean_or_scale_map = parseMeanOrScaleString(mean_scale);

    for (auto&& [layer_name, mean_or_scale] : mean_or_scale_map) {
        if (!layer_name.empty()) {
            // Add an explicit reference. Lambda expressions in C++17 cannot capture structured bindings.
            const auto& layer_name_ref = layer_name;
            auto required_input_it = std::find_if(inputs_info.begin(), inputs_info.end(),
                                                  [&layer_name_ref](const ov::Output<const ov::Node>& item) {
                                                      return item.get_any_name() == layer_name_ref;
                                                  });
            if (required_input_it != inputs_info.end()) {
                result[std::distance(inputs_info.begin(), required_input_it)] = mean_or_scale;
            } else {
                throw std::logic_error(std::string("Input with name '") + layer_name + "' doesn't exist.");
            }
        } else {
            for (size_t idx = 0; idx < inputs_info.size(); ++idx) {
                result[idx] = mean_or_scale;
            }
        }
    }

    return result;
}

using RegexPtr = std::unique_ptr<std::regex>;
std::map<RegexPtr, ov::Layout> parseLayoutRegex(std::string layouts) {
    std::map<std::string, std::string> input_output_layouts = parseArgMap(std::move(layouts));

    std::map<RegexPtr, ov::Layout> out;
    for (const auto& input_output_layout : input_output_layouts) {
        auto [name, value] = input_output_layout;
        if (value.empty()) {
            if (name.empty()) {
                throw std::runtime_error("Can't parse layouts string \"" + layouts +
                                         "\" into valid \"input:layout;input:layout\" pairs");
            }
            // there is no value only name, thus we consider input/output name as "any" and
            // apply layout value as the parsed name
            out.emplace(std::make_unique<std::regex>(".*"), name);
            continue;
        }
        std::string valid_regex_str = name.empty() ? ".*" : "^" + name + "$";
        out.emplace(std::make_unique<std::regex>(std::move(valid_regex_str)), std::move(value));
    }
    return out;
}

template <class T>
std::optional<T> getRegexSubstitutionIfExist(const std::string& haystack, const std::map<RegexPtr, T>& substitutions) {
    for (const auto& s : substitutions) {
        if (std::regex_search(haystack, *s.first)) {
            return {s.second};
        }
    }
    return {};
}
//
// File utils
//

bool hasLoadableExt(const std::string& network_path) {
    static const std::array<const char*, 5> ext_support_table{"xml", "onnx", "pdmodel", "pb", "tflite"};
    return std::any_of(ext_support_table.begin(), ext_support_table.end(), [&network_path](const char* ext) {
        static constexpr const auto dot_symbol = '.';
        auto pos = network_path.rfind(dot_symbol);
        std::string ext_name = {};
        if (pos != std::string::npos)
            ext_name = network_path.substr(pos + 1);
        return strEq(ext_name, ext);
    });
}

std::string cleanName(std::string&& name) {
    std::replace_if(
            name.begin(), name.end(),
            [](unsigned char c) {
                return !std::isalnum(c);
            },
            '_');
    return std::move(name);
}

ov::Tensor loadImage(const ov::element::Type& precision, const ov::Shape& shape, const ov::Layout& layout,
                     const std::string& filePath, const std::string& colorFormat) {
    const auto frame = cv::imread(filePath, cv::IMREAD_COLOR);
    OPENVINO_ASSERT(!frame.empty(), "Failed to open input image file ", filePath);

    const ov::Tensor tensor(precision, shape);

    cvToOV(frame, tensor, shape, layout, colorFormat);

    return tensor;
}

ov::Tensor loadBinary(const ov::element::Type& modelPrecision, const ov::Shape& shape, const ov::Layout& layout,
                      const std::string& filePath, const ov::element::Type& dataPrecision) {
    std::ifstream binaryFile(filePath, std::ios_base::binary | std::ios_base::ate);
    OPENVINO_ASSERT(binaryFile, "Failed to open input binary file: ", filePath);
    const auto fileSize = binaryFile.tellg();
    binaryFile.seekg(0, std::ios_base::beg);
    OPENVINO_ASSERT(binaryFile.good(), "While reading a file an error is encountered");
    const size_t fileBytes = static_cast<size_t>(fileSize);
    ov::Tensor requestedTensor(modelPrecision, shape);
    const size_t reqTensorBytes = static_cast<size_t>(requestedTensor.get_byte_size());

    if (dataPrecision != modelPrecision && dataPrecision != ov::element::Type_t::undefined) {
        std::cout << "Converting " << filePath << " input from " << dataPrecision << " to " << modelPrecision
                  << std::endl;
        const ov::Tensor inputTensor(dataPrecision, shape);
        if (fileBytes == inputTensor.get_byte_size()) {
            binaryFile.read(reinterpret_cast<char*>(inputTensor.data()), static_cast<std::streamsize>(fileBytes));
            npu::utils::convertTensorPrecision(inputTensor, requestedTensor);
        } else {
            std::cout << "File contains " << fileBytes
                      << " bytes, but it expected to be: " << inputTensor.get_byte_size()
                      << " while converting precision from " << dataPrecision << " to " << modelPrecision
                      << ". Check whether it is possible to batch loading " << std::endl;
            OPENVINO_ASSERT(ov::layout::has_batch(layout),
                            "Input layout has no batch dimenstion: ", layout.to_string());
            size_t N = shape[ov::layout::batch_idx(layout)];
            OPENVINO_ASSERT(fileBytes * N == inputTensor.get_byte_size(), "File contains ", fileBytes, " bytes, but ",
                            inputTensor.get_byte_size() * N, " total in batch size ", N,
                            " expected while converting precision from ", dataPrecision, " to ", modelPrecision);
            ov::Shape debatchedInputTensorShape(shape);
            debatchedInputTensorShape[ov::layout::batch_idx(layout)] = 1;
            const ov::Tensor inputDebatchedTensor(dataPrecision, debatchedInputTensorShape);
            binaryFile.read(reinterpret_cast<char*>(inputDebatchedTensor.data()),
                            static_cast<std::streamsize>(fileBytes));
            const ov::Tensor convertedPrecisionTensor(modelPrecision, debatchedInputTensorShape);
            npu::utils::convertTensorPrecision(inputDebatchedTensor, convertedPrecisionTensor);
            std::list<ov::Tensor> tensorsToJoin;
            std::generate_n(std::back_inserter(tensorsToJoin), N, [&convertedPrecisionTensor]() {
                return convertedPrecisionTensor;
            });
            requestedTensor = npu::utils::joinTensors(tensorsToJoin, layout);
        }

    } else {
        if (fileBytes == reqTensorBytes) {
            binaryFile.read(reinterpret_cast<char*>(requestedTensor.data()),
                            static_cast<std::streamsize>(reqTensorBytes));
        } else {
            std::cout << "File contains " << fileBytes << " bytes, but it expected to be: " << reqTensorBytes
                      << " when datatypes match. "
                      << ". Check whether it is possible to batch loading " << std::endl;
            OPENVINO_ASSERT(ov::layout::has_batch(layout),
                            "Input layout has no batch dimenstion: ", layout.to_string());
            size_t N = shape[ov::layout::batch_idx(layout)];
            OPENVINO_ASSERT(fileBytes * N == reqTensorBytes, "File contains ", fileBytes, " bytes, but ",
                            reqTensorBytes, " in batch size ", N, " expected");

            // duplicate a binary into tensor memory if the tensor batched
            for (size_t n = 0; n < N; ++n) {
                binaryFile.seekg(0, std::ios_base::beg);
                binaryFile.read(reinterpret_cast<char*>(requestedTensor.data()) + fileBytes * n,
                                static_cast<std::streamsize>(fileBytes));
            }
        }
    }

    return requestedTensor;
}

/**
 * @brief Loads the contents of a locally stored file inside an OpenVINO tensor intended to be used as input in the
 * context of the application.
 * @details The data being loaded can either be an image or a binary file, the switch between these cases can be
 * performed by setting the "img_as_bin" flag accordingly. If an image is being loaded, the OpenCV library is deployed
 * for reading followed by a conversion to the OpenVINO format. If a binary file is loaded, the content's type is
 * converted from "dataPrecision" to "modelPrecision" before constructing the tensor.
 *
 * @param modelPrecision The precision accepted by the model's input
 * @param shape The shape accepted by the model's input
 * @param layout The layout used by the model's input
 * @param filePath Indicates the location of the file to be loaded
 * @param colorFormat Indicates the color format only in the case when an image is being loaded.
 * @param dataPrecision Indicates the precision used by the data found within the binary file.
 * @return The tensor containing the loaded data.
 */
ov::Tensor loadInput(const ov::element::Type& modelPrecision, const ov::Shape& shape, const ov::Layout& layout,
                     const std::string& filePath, const std::string& colorFormat,
                     const ov::element::Type& dataPrecision = ov::element::Type_t::undefined) {
    if (isImage(shape, layout) && !FLAGS_img_as_bin) {
        return loadImage(modelPrecision, shape, layout, filePath, colorFormat);
    } else {
        return loadBinary(modelPrecision, shape, layout, filePath, dataPrecision);
    }
}

ov::Tensor loadTensor(const ov::element::Type& precision, const ov::Shape& shape, const std::string& filePath) {
    const ov::Tensor tensor(precision, shape);

    std::ifstream file(filePath, std::ios_base::in | std::ios_base::binary);
    OPENVINO_ASSERT(file.is_open(), "Can't open file ", filePath, " for read");

    const auto dataBuffer = reinterpret_cast<char*>(tensor.data());
    file.read(dataBuffer, static_cast<std::streamsize>(tensor.get_byte_size()));

    return tensor;
}

void dumpTensor(const ov::Tensor& tensor, const std::string& filePath) {
    std::ofstream file(filePath, std::ios_base::out | std::ios_base::binary);
    OPENVINO_ASSERT(file.is_open(), "Can't open file ", filePath, " for write");

    const auto dataBuffer = reinterpret_cast<char*>(tensor.data());
    file.write(dataBuffer, static_cast<std::streamsize>(tensor.get_byte_size()));
}

std::map<std::string, std::string> parseConfigFile() {
    std::map<std::string, std::string> config;

    std::ifstream file(FLAGS_config);
    OPENVINO_ASSERT(file.is_open(), "Can't open file ", FLAGS_config, " for read");

    std::string option;
    while (std::getline(file, option)) {
        if (option.empty() || option[0] == '#') {
            continue;
        }
        size_t spacePos = option.find_first_of(" \t\n\r");
        OPENVINO_ASSERT(spacePos != std::string::npos,
                        "Invalid config parameter format. Space separator required here: ", option);
        std::string key, value;
        if (spacePos != std::string::npos) {
            key = option.substr(0, spacePos);
            size_t valueStart = option.find_first_not_of(" \t\n\r", spacePos);
            OPENVINO_ASSERT(valueStart != std::string::npos,
                            "An invalid config parameter value detected, it mustn't be empty: ", option);
            size_t valueEnd = option.find_last_not_of(" \t\n\r");
            value = option.substr(valueStart, valueEnd - valueStart + 1);
            config[key] = value;
        }
    }

    return config;
}

// This function formats performance counters in a same way as benchmark_app -pc does.
// It is a copy-paste from $OPENVINO_HOME/samples/cpp/common/utils/include/samples/common.hpp
using ProfVec = std::vector<ov::ProfilingInfo>;
static void printPerformanceCounts(ProfVec performanceData, std::ostream& stream, std::string deviceName,
                                   bool bshowHeader = true) {
    std::chrono::microseconds totalTime = std::chrono::microseconds::zero();
    // Print performance counts
    if (bshowHeader) {
        stream << std::endl << "performance counts:" << std::endl << std::endl;
    }
    std::ios::fmtflags fmt(std::cout.flags());
    for (const auto& it : performanceData) {
        std::string toPrint(it.node_name);
        const int maxLayerName = 30;

        if (it.node_name.length() >= maxLayerName) {
            toPrint = it.node_name.substr(0, maxLayerName - 4);
            toPrint += "...";
        }

        stream << std::setw(maxLayerName) << std::left << toPrint;
        switch (it.status) {
        case ov::ProfilingInfo::Status::EXECUTED:
            stream << std::setw(15) << std::left << "EXECUTED";
            break;
        case ov::ProfilingInfo::Status::NOT_RUN:
            stream << std::setw(15) << std::left << "NOT_RUN";
            break;
        case ov::ProfilingInfo::Status::OPTIMIZED_OUT:
            stream << std::setw(15) << std::left << "OPTIMIZED_OUT";
            break;
        }
        stream << std::setw(30) << std::left << "layerType: " + std::string(it.node_type) + " ";
        stream << std::setw(20) << std::left << "realTime: " + std::to_string(it.real_time.count());
        stream << std::setw(20) << std::left << "cpu: " + std::to_string(it.cpu_time.count());
        stream << " execType: " << it.exec_type << std::endl;
        if (it.real_time.count() > 0) {
            totalTime += it.real_time;
        }
    }
    stream << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime.count()) << " microseconds"
           << std::endl;
    std::cout << std::endl;
    std::cout << "Full device name: " << deviceName << std::endl;
    std::cout << std::endl;
    std::cout.flags(fmt);
}

bool checkBBoxOutputs(std::vector<utils::BoundingBox>& actualOutput, std::vector<utils::BoundingBox>& refOutput,
                      const size_t imgWidth, const size_t imgHeight, const float boxTolerance,
                      const float probTolerance) {
    std::cout << "Ref Top:" << std::endl;
    for (size_t i = 0; i < refOutput.size(); ++i) {
        const auto& bb = refOutput[i];
        std::cout << i << " : " << bb.idx << " : [(" << bb.left << " " << bb.top << "), (" << bb.right << " "
                  << bb.bottom << ")] : " << bb.prob * 100 << "%" << std::endl;
    }

    std::cout << "Actual top:" << std::endl;
    for (size_t i = 0; i < actualOutput.size(); ++i) {
        const auto& bb = actualOutput[i];
        std::cout << i << " : " << bb.idx << " : [(" << bb.left << " " << bb.top << "), (" << bb.right << " "
                  << bb.bottom << ")] : " << bb.prob * 100 << "%" << std::endl;
    }

    for (const auto& refBB : refOutput) {
        bool found = false;

        float maxBoxError = 0.0f;
        float maxProbError = 0.0f;

        for (const auto& actualBB : actualOutput) {
            if (actualBB.idx != refBB.idx) {
                continue;
            }

            const utils::Box actualBox{actualBB.left / imgWidth, actualBB.top / imgHeight,
                                       (actualBB.right - actualBB.left) / imgWidth,
                                       (actualBB.bottom - actualBB.top) / imgHeight};
            const utils::Box refBox{refBB.left / imgWidth, refBB.top / imgHeight, (refBB.right - refBB.left) / imgWidth,
                                    (refBB.bottom - refBB.top) / imgHeight};

            const auto boxIntersection = boxIntersectionOverUnion(actualBox, refBox);
            const auto boxError = 1.0f - boxIntersection;
            maxBoxError = std::max(maxBoxError, boxError);

            const auto probError = std::fabs(actualBB.prob - refBB.prob);
            maxProbError = std::max(maxProbError, probError);

            if (boxError > boxTolerance) {
                continue;
            }

            if (probError > probTolerance) {
                continue;
            }

            found = true;
            break;
        }
        if (!found) {
            std::cout << "maxBoxError=" << maxBoxError << " "
                      << "maxProbError=" << maxProbError << std::endl;
            return false;
        }
    }
    return true;
}

//
// Classification mode
//

std::vector<std::pair<int, float>> parseClassification(const float* dataBuffer, size_t dataBufferElementsCount) {
    OPENVINO_ASSERT(dataBuffer != nullptr, "Received a tensor with no allocated buffer");

    std::vector<std::pair<int, float>> res(dataBufferElementsCount);
    for (size_t i = 0; i < dataBufferElementsCount; ++i) {
        res[i].first = static_cast<int>(i);
        res[i].second = dataBuffer[i];
    }

    std::sort(res.begin(), res.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second;
    });

    return res;
}

std::vector<std::vector<std::pair<int, float>>> parseClassificationBatch(const ov::Tensor& tensor, size_t batch_size) {
    OPENVINO_ASSERT(batch_size, "batch_size can't be 0");
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::Type_t::f32,
                    "Unsupported precision: ", tensor.get_element_type().get_type_name());

    std::vector<std::vector<std::pair<int, float>>> ret;

    const float* dataBuffer = tensor.data<const float>();
    OPENVINO_ASSERT(dataBuffer != nullptr, "Received a tensor with no allocated buffer");

    size_t batch_bundle_size = tensor.get_size() / batch_size;
    OPENVINO_ASSERT(!(tensor.get_size() % batch_bundle_size),
                    "Tensor is a not batched tensor! Size: ", tensor.get_size(),
                    " can't be batched on a batch size: ", batch_size, " properly");

    size_t i = 0;
    for (; i < tensor.get_size(); i += batch_bundle_size) {
        if (batch_size != 1) {
            std::cout << "restore tensor from data bundle: (" << i << "/" << tensor.get_size() << " bytes)"
                      << std::endl;
        }
        ret.push_back(parseClassification(dataBuffer + i, batch_bundle_size));
    }

    OPENVINO_ASSERT(i == tensor.get_size());
    return ret;
}

bool testClassification(const TensorMap& outputs, const TensorMap& references, size_t batch_size = 1) {
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(outputs.size() == references.size());

    const ov::Tensor outputFP32 = npu::utils::toFP32(outputs.begin()->second);
    const ov::Tensor referenceFP32 = npu::utils::toFP32(references.begin()->second);

    OPENVINO_ASSERT(outputFP32.get_element_type() == referenceFP32.get_element_type());
    OPENVINO_ASSERT(outputFP32.get_shape() == referenceFP32.get_shape());
    OPENVINO_ASSERT(referenceFP32.get_element_type() == ov::element::Type_t::f32);

    auto probsBatch = parseClassificationBatch(outputFP32, batch_size);
    auto refProbsBatch = parseClassificationBatch(referenceFP32, batch_size);
    OPENVINO_ASSERT(refProbsBatch.size() == probsBatch.size(),
                    "Incorrect batch size of both output tensor: ", probsBatch.size(),
                    " and reference tensor: ", refProbsBatch.size(), ". Expected: ", batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        OPENVINO_ASSERT(probsBatch[i].size() == refProbsBatch[i].size(),
                        "Incorrect size of referenced tensor in batch bundle number: (", i, "/", batch_size, ")",
                        ". Expected size: ", probsBatch[i].size(), ", got: ", refProbsBatch[i].size());
        OPENVINO_ASSERT(refProbsBatch[i].size() >= FLAGS_top_k);
        refProbsBatch[i].resize(FLAGS_top_k);
    }

    bool result = true;
    for (size_t i = 0; i < probsBatch.size(); i++) {
        if (batch_size != 1) {
            std::cout << "Check tensor bundle: (" << i << "/" << batch_size << " batch)" << std::endl;
        }
        auto probs = probsBatch[i];
        const auto& refs = refProbsBatch[i];
        OPENVINO_ASSERT(probs.size() >= FLAGS_top_k);
        probs.resize(FLAGS_top_k);

        std::cout << "Actual top:" << std::endl;
        for (size_t j = 0; j < probs.size(); ++j) {
            std::cout << "    " << j << " : " << probs[j].first << " : " << probs[j].second << std::endl;
        }

        std::cout << "Ref Top:" << std::endl;
        for (size_t j = 0; j < refs.size(); ++j) {
            std::cout << "    " << j << " : " << refs[j].first << " : " << refs[j].second << std::endl;
        }

        for (const auto& refElem : refs) {
            const auto actualIt =
                    std::find_if(probs.cbegin(), probs.cend(), [&refElem](const std::pair<int, float>& arg) {
                        return refElem.first == arg.first;
                    });
            if (actualIt == probs.end()) {
                std::cout << "Ref result " << refElem.first << " was not found in actual results" << std::endl;
                result = result && false;
                continue;
            }

            const auto& actualElem = *actualIt;

            if (refElem.second > actualElem.second) {
                const auto probDiff = std::fabs(refElem.second - actualElem.second);
                if (probDiff > FLAGS_prob_tolerance) {
                    std::cout << "Probability value mismatch for " << refElem.first << " : " << refElem.second << " vs "
                              << actualElem.second;
                    result = result && false;
                }
            }
        }
    }

    return result;
}

//
// RAW mode
//

bool compareTensors(const ov::Tensor& output, const ov::Tensor& reference) {
    if (output.get_shape() != reference.get_shape()) {
        std::cout << "Output and reference tensors have different shapes" << std::endl;
        return false;
    }

    const ov::Tensor outputFP32 = npu::utils::toFP32(output);
    const ov::Tensor referenceFP32 = npu::utils::toFP32(reference);

    const auto outputBuffer = outputFP32.data<const float>();
    const auto referenceBuffer = referenceFP32.data<const float>();

    const auto totalCount = referenceFP32.get_size();
    const auto printCount = std::min<size_t>(totalCount, 10);

    for (size_t i = 0; i < totalCount; ++i) {
        const auto referenceValue = referenceBuffer[i];
        const auto outputValue = outputBuffer[i];
        const auto absDiff = std::fabs(referenceValue - outputValue);

        if (i < printCount) {
            std::cout << "        " << i << " :"
                      << " ref : " << std::setw(10) << referenceValue << " output : " << std::setw(10) << outputValue
                      << " absdiff : " << std::setw(10) << absDiff << std::endl;
        }

        if (absDiff > FLAGS_raw_tolerance) {
            std::cout << "Absolute difference between output value " << outputValue << " and reference value "
                      << referenceValue << " at index " << i << " larger then tolerance" << std::endl;
            return false;
        }
    }

    return true;
}

bool testRAW(const TensorMap& outputTensors, const TensorMap& referenceTensors, size_t batch_size = 1) {
    if (batch_size != 1) {
        throw std::runtime_error(
                "The testcase 'raw' doesn't support any `override_model_batch_size` values besides 1 yet");
    }

    if (outputTensors.size() != referenceTensors.size()) {
        std::cout << "The number of predicted outputs differ from the number of references" << std::endl;
        return false;
    }

    for (const auto& [tensorName, outputTensor] : outputTensors) {
        auto referenceTensorIterator = referenceTensors.find(tensorName);
        OPENVINO_ASSERT(referenceTensorIterator != referenceTensors.end());

        std::cout << "Compare " << tensorName << " with reference" << std::endl;
        if (!compareTensors(outputTensor, referenceTensorIterator->second)) {
            return false;
        }
    }

    return true;
}

//
// Cosine-Similarity mode
// (using 'cosim_threshold' flag, with expected value in range [0.0 -> 1.0])
// e.g. '--mode cosim --cosim_threshold 0.98'
//

bool compareCoSim(const ov::Tensor& output, const ov::Tensor& reference) {
    if (output.get_shape() != reference.get_shape()) {
        std::cout << "Actual and reference blobs has different shape" << std::endl;
        return false;
    }

    const ov::Tensor outputFP32 = npu::utils::toFP32(output);
    const ov::Tensor referenceFP32 = npu::utils::toFP32(reference);

    const auto outputBuffer = outputFP32.data<const float>();
    const auto referenceBuffer = referenceFP32.data<const float>();

    const auto size = referenceFP32.get_size();

    double numr = 0.0, denA = 0.0, denB = 0.0;
    for (size_t i = 0; i < size; ++i) {
        numr += outputBuffer[i] * referenceBuffer[i];
        denA += outputBuffer[i] * outputBuffer[i];
        denB += referenceBuffer[i] * referenceBuffer[i];
    }

    if (denA == 0 || denB == 0) {
        std::cout << "Div by ZERO. Cannot compute CoSim metric" << std::endl;
        return false;
    }

    const auto similarity = numr / (sqrt(denA) * sqrt(denB));
    const double eps = 0.0000001;
    // Some experiments revealed that when applying the CoSim metric to large buffers it could provide
    // similarity values that are outside the [-1:1] interval due the big number of operations done on
    // floating point value. A small epsilon value was added to extend the interval to [-(1+eps):1+eps]
    // to ensure that the above check is not failing.
    if (similarity > (1.0 + eps) || similarity < -(1.0 + eps)) {
        std::cout << "Invalid result " << similarity << " (valid range [-1 : +1])" << std::endl;
        return false;
    }

    std::cout << "Cosine similarity : " << similarity * 100 << "%" << std::endl;
    return similarity > FLAGS_cosim_threshold;
}

bool testCoSim(const TensorMap& outputs, const TensorMap& references, size_t batch_size = 1) {
    if (batch_size != 1) {
        throw std::runtime_error(
                "The testcase 'testCoSim' doesn't support any `override_model_batch_size` values besides 1 yet");
    }

    if (outputs.size() != references.size()) {
        std::cout << "The outputs and references differ in the number of tensors" << std::endl;
        return false;
    }

    for (const auto& [tensorName, output] : outputs) {
        auto referencesIterator = references.find(tensorName);
        OPENVINO_ASSERT(referencesIterator != references.end());

        std::cout << "Compare " << tensorName << " with reference" << std::endl;
        if (!compareCoSim(output, referencesIterator->second)) {
            return false;
        }
    }

    return true;
}

//
// Relative Root Mean Squared Error mode
// (using 'rrmse_loss_threshold' flag, with expected value in range [0.0 -> infinity))
// e.g. '--mode rrmse --rrmse_loss_threshold 0.1'
//

bool computeRRMSE(const ov::Tensor& output, const ov::Tensor& reference) {
    if (output.get_shape() != reference.get_shape()) {
        std::cout << "Output and reference tensors have different shapes" << std::endl;
        return false;
    }

    const ov::Tensor outputFP32 = npu::utils::toFP32(output);
    const ov::Tensor referenceFP32 = npu::utils::toFP32(reference);

    const auto outputBuffer = outputFP32.data<const float>();
    const auto referenceBuffer = referenceFP32.data<const float>();

    const auto size = referenceFP32.get_size();

    double error = 0, sum = 0, diff;
    for (size_t i = 0; i < size; ++i) {
        diff = (outputBuffer[i] - referenceBuffer[i]);
        sum += (outputBuffer[i] * outputBuffer[i]);
        error += (diff * diff);
    }

    if (sum == 0) {
        if (error <= std::numeric_limits<double>::epsilon()) {
            std::cout << "The results perfectly match (error = 0). RRMSE loss could not be computed" << std::endl;
            return true;
        }

        std::cout << "Div by ZERO (Actual is the Zero Tensor). Cannot compute RRMSE loss" << std::endl;
        return false;
    }

    double rrmseLoss = sqrt(error / sum);

    std::cout << "RRMSE loss : " << std::fixed << std::setprecision(4) << rrmseLoss
              << "   RRMSE threshold : " << FLAGS_rrmse_loss_threshold << std::endl;
    return rrmseLoss <= FLAGS_rrmse_loss_threshold;
}

bool testRRMSE(const TensorMap& outputs, const TensorMap& references, size_t batch_size = 1) {
    if (batch_size != 1) {
        throw std::runtime_error(
                "The testcase 'rrmse' doesn't support any `override_model_batch_size` values besides 1 yet");
    }

    if (outputs.size() != references.size()) {
        std::cout << "Actual and reference has different number of output blobs" << std::endl;
        return false;
    }

    for (const auto& [tensorName, output] : outputs) {
        auto referencesIterator = references.find(tensorName);
        OPENVINO_ASSERT(referencesIterator != references.end());

        std::cout << "Compare " << tensorName << " with reference" << std::endl;
        if (!computeRRMSE(output, referencesIterator->second)) {
            return false;
        }
    }

    return true;
}

//
// Normalized Mean Squared Error mode
// (using 'nrmse_loss_threshold' flag, with expected value in range [0.0 -> infinity))
// e.g. '--mode nrmse --nrmse_loss_threshold 0.01'
//

bool computeNRMSE(const ov::Tensor& output, const ov::Tensor& reference) {
    if (output.get_shape() != reference.get_shape()) {
        std::cout << "Output and reference tensors have different shapes" << std::endl;
        return false;
    }

    const auto size = reference.get_size();

    if (size == 0) {
        std::cout << "Empty output and reference tensors, NRMSE loss set to 0" << std::endl;
        return true;
    }

    const ov::Tensor outputFP32 = npu::utils::toFP32(output);
    const ov::Tensor referenceFP32 = npu::utils::toFP32(reference);

    const auto outputBuffer = outputFP32.data<const float>();
    const auto referenceBuffer = referenceFP32.data<const float>();

    double error = 0;
    float maxOutput = 0, maxReference = 0, minOutput = 0, minReference = 0;
    for (size_t i = 0; i < size; ++i) {
        const auto diff = outputBuffer[i] - referenceBuffer[i];
        error += diff * diff;
        maxOutput = std::max(outputBuffer[i], maxOutput);
        maxReference = std::max(referenceBuffer[i], maxReference);
        minOutput = std::min(outputBuffer[i], minOutput);
        minReference = std::min(referenceBuffer[i], minReference);
    }

    double nrmseLoss =
            sqrt(error / size) / std::max(0.001f, std::max(maxOutput - minOutput, maxReference - minReference));

    std::cout << "NRMSE loss : " << std::fixed << std::setprecision(4) << nrmseLoss
              << "   NRMSE threshold : " << FLAGS_nrmse_loss_threshold << std::endl;
    return nrmseLoss <= FLAGS_nrmse_loss_threshold;
}

bool testNRMSE(const TensorMap& outputs, const TensorMap& references, size_t batch_size = 1) {
    if (batch_size != 1) {
        throw std::runtime_error(
                "The testcase 'nrmse' doesn't support any `override_model_batch_size` values besides 1 yet");
    }

    if (outputs.size() != references.size()) {
        std::cout << "Actual and reference has different number of output blobs" << std::endl;
        return false;
    }

    for (const auto& [tensorName, output] : outputs) {
        auto referencesIterator = references.find(tensorName);
        OPENVINO_ASSERT(referencesIterator != references.end());

        std::cout << "Compare " << tensorName << " with reference" << std::endl;
        if (!computeNRMSE(output, referencesIterator->second)) {
            return false;
        }
    }

    return true;
}

//
// PSNR mode
// using psnr_reference and psnr_tolerance flags for validation
// e.g. '--mode psnr --psnr_reference <value> --psnr_tolerance <value>'
// Direction of metric’s growth is higher-better. If the images are identical, the PSNR is infinite.
//

bool testPSNR(const TensorMap& outputs, const TensorMap& references, const int dstHeight, const int dstWidth,
              size_t batch_size = 1) {
    if (batch_size != 1) {
        throw std::runtime_error(
                "The testcase 'psnr' doesn't support any `override_model_batch_size` values besides 1 yet");
    }
    OPENVINO_ASSERT(outputs.size() == references.size(),
                    "Mismatch between the number of model outputs and the number of references");

    int scaleBorder = FLAGS_scale_border;
    bool normalizedImage = FLAGS_normalized_image;

    auto refOutput = npu::utils::parseTensorsAsFP32(references);
    auto actOutput = npu::utils::parseTensorsAsFP32(outputs);

    auto result = utils::runPSNRMetric(actOutput, refOutput, dstHeight, dstWidth, scaleBorder, normalizedImage);

    if (FLAGS_psnr_reference - result > FLAGS_psnr_tolerance) {
        std::cout << "Absolute difference between actual value " << result << " and reference value "
                  << FLAGS_psnr_reference << " larger then tolerance " << FLAGS_psnr_tolerance << std::endl;
        return false;
    }

    return true;
}

static void printPerformanceCountsAndLatency(size_t numberOfTestCase, const ProfVec& profilingData,
                                             std::chrono::duration<double, std::milli> duration) {
    auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    if (!profilingData.empty()) {
        std::cout << "Performance counts for " << numberOfTestCase << "-th infer request:" << std::endl;
        printPerformanceCounts(profilingData, std::cout, FLAGS_device, false);
    }

    std::cout << "Latency: " << std::fixed << std::setprecision(2) << durationMs.count() << " ms" << std::endl;
}

bool compare_mean_IoU(std::vector<std::pair<bool, float>> iou, float semSegThreshold, uint32_t classes) {
    float threshold = semSegThreshold * 100;
    float ma = 0.0f;
    bool stateValue = true;

    if (FLAGS_sem_seg_ignore_label != std::numeric_limits<uint32_t>::max()) {
        classes--;
    }

    size_t numberOfLabeledClasses = 0;
    for (size_t i = 0; i < classes; i++) {
        if (iou[i].first) {
            numberOfLabeledClasses++;
            if (FLAGS_dataset == "camVid12") {
                std::cout << "mean_iou@" << camVid12[i].c_str() << ": " << std::fixed << std::setprecision(2)
                          << iou[i].second << "%" << std::endl;
            } else {
                std::cout << "mean_iou@class" << i << ": " << std::fixed << std::setprecision(2) << iou[i].second << "%"
                          << std::endl;
            }
            if (iou[i].second < threshold) {
                std::cout << "Threshold smaller than " << threshold << "%" << std::endl;
                stateValue = false;
            }
            ma += iou[i].second;
        } else {
            std::cout << "mean_iou@class" << i << ": no pixels labeled." << std::endl;
        }
    }
    std::cout << "mean_iou@:mean " << std::fixed << std::setprecision(2) << (ma / numberOfLabeledClasses) << "%"
              << std::endl;

    return stateValue;
}

// CVS-143420 Allow TEMPLATE plugin to be used instead of CPU to avoid accuracy regressions
#ifdef _WIN32
const char TEMPLATE_LIB[] = "openvino_template_plugin.dll";
#else
const char TEMPLATE_LIB[] = "libopenvino_template_plugin.so";
#endif

void setupOVCore(ov::Core& core) {
    auto flagDevice = FLAGS_device;

    if (FLAGS_device == "TEMPLATE") {
        core.register_plugin(TEMPLATE_LIB, FLAGS_device);
    }

    if (!FLAGS_log_level.empty()) {
        core.set_property(flagDevice, {{ov::log::level.name(), FLAGS_log_level}});
    }

    if (FLAGS_device == "CPU") {
        core.set_property(flagDevice, {{"LP_TRANSFORMS_MODE", "NO"}});
    }

    if (FLAGS_pc) {
        core.set_property(flagDevice, {{ov::enable_profiling.name(), true}});
    }

    if (!FLAGS_config.empty()) {
        const auto configs = parseConfigFile();
        core.set_property(flagDevice, {configs.begin(), configs.end()});
    }
}

void nameIOTensors(std::shared_ptr<ov::Model> model) {
    auto inputInfo = model->inputs();
    for (std::size_t id = 0ul; id < inputInfo.size(); ++id) {
        auto ii = inputInfo[id];
        if (ii.get_names().empty()) {
            ii.add_names({"input_" + std::to_string(ii.get_index()) + "_" + std::to_string(id)});
        }
    }

    auto outputInfo = model->outputs();
    for (std::size_t id = 0ul; id < outputInfo.size(); ++id) {
        auto oi = outputInfo[id];
        if (oi.get_names().empty()) {
            oi.add_names({"output_" + std::to_string(oi.get_index()) + "_" + std::to_string(id)});
        }
    }
}

std::pair<TensorMap, ProfVec> runInfer(ov::InferRequest& inferRequest, ov::CompiledModel& compiledModel,
                                       const TensorMap& inputs, const std::vector<std::string>& dumpedInputsPaths) {
    for (const auto& [tensorName, tensor] : inputs) {
        inferRequest.set_tensor(tensorName, tensor);
    }

    inferRequest.infer();

    TensorMap out;
    for (const auto& outputInfo : compiledModel.outputs()) {
        const std::string layer_name = outputInfo.get_any_name();
        out.insert({layer_name, inferRequest.get_tensor(layer_name)});
    }

    ProfVec profData{};

    if (FLAGS_pc) {
        profData = inferRequest.get_profiling_info();
    }

    return std::make_pair(out, profData);
}

// FIXME: User must provide layout explicitly.
// No "default" layout for IRv11 models.
static ov::Layout getLayoutByRank(const size_t rank) {
    switch (rank) {
    case 0:
        return ov::Layout::scalar();
    case 1:
        return ov::Layout("C");
    case 2:
        return ov::Layout("NC");
    case 3:
        return ov::Layout("CHW");
    case 4:
        return ov::Layout("NCHW");
    case 5:
        return ov::Layout("NCDHW");
    }
    throw std::logic_error("Failed to get layout for rank equal to " + std::to_string(rank));
}

static std::string toString(const std::vector<size_t>& vec) {
    std::stringstream ss;
    if (!vec.empty()) {
        ss << "[";
        for (size_t i = 0; i < vec.size() - 1; ++i) {
            ss << vec[i] << ",";
        }
        ss << vec[vec.size() - 1];
        ss << "]";
    } else {
        ss << "SCALAR";
    }
    return ss.str();
}

bool testSSDDetection(const TensorMap& outputs, const TensorMap& references,
                      const TensorDescriptorMap& inputDescriptors, size_t batch_size = 1) {
    if (batch_size != 1) {
        throw std::runtime_error(
                "The testcase 'ssd' doesn't support any `override_model_batch_size` values besides 1 yet");
    }

    OPENVINO_ASSERT(outputs.size() == 1 && references.size() == 1);
    OPENVINO_ASSERT(!inputDescriptors.empty(), "No input descriptors received");

    const ov::Tensor& output = outputs.begin()->second;
    const ov::Tensor& reference = references.begin()->second;
    const TensorDescriptor& inputDescriptor = inputDescriptors.begin()->second;

    const auto imgWidth = inputDescriptor.dataShape.at(ov::layout::width_idx(inputDescriptor.layout));
    const auto imgHeight = inputDescriptor.dataShape.at(ov::layout::height_idx(inputDescriptor.layout));

    auto confThresh = FLAGS_confidence_threshold;
    auto probTolerance = FLAGS_prob_tolerance;
    auto boxTolerance = FLAGS_box_tolerance;

    auto parsedOutput = utils::parseSSDOutput(output, imgWidth, imgHeight, static_cast<float>(confThresh));
    auto parsedReference = utils::parseSSDOutput(reference, imgWidth, imgHeight, static_cast<float>(confThresh));

    auto result = checkBBoxOutputs(parsedOutput, parsedReference, imgWidth, imgHeight, static_cast<float>(boxTolerance),
                                   static_cast<float>(probTolerance));

    return result;
}

//
// Yolo V2 mode
//
bool testYoloV2(const TensorMap& outputs, const TensorMap& references, const TensorDescriptorMap& inputDescriptors,
                size_t batch_size = 1) {
    if (batch_size != 1) {
        throw std::runtime_error(
                "The testcase 'yolo_v2' doesn't support any `override_model_batch_size` values besides 1 yet");
    }
    OPENVINO_ASSERT(inputDescriptors.size() == 1, "The YOLO v2 model accepts only a single input");
    OPENVINO_ASSERT(outputs.size() == 1, "The YOLO v2 model a single output");
    OPENVINO_ASSERT(outputs.size() == references.size(),
                    "Mismatch between the number of model outputs and the number of references");
    const ov::Tensor& output = outputs.begin()->second;
    const ov::Tensor& reference = references.begin()->second;

    const TensorDescriptor& inputDescriptor = inputDescriptors.begin()->second;

    const auto imgWidth = inputDescriptor.dataShape.at(ov::layout::width_idx(inputDescriptor.layout));
    const auto imgHeight = inputDescriptor.dataShape.at(ov::layout::height_idx(inputDescriptor.layout));
    double confThresh = FLAGS_confidence_threshold;
    double probTolerance = FLAGS_prob_tolerance;
    double boxTolerance = FLAGS_box_tolerance;
    bool isTiny = FLAGS_is_tiny_yolo;

    auto parsedOutput = utils::parseYoloOutput(npu::utils::toFP32(output), imgWidth, imgHeight,
                                               static_cast<float>(confThresh), isTiny);
    auto parsedReference = utils::parseYoloOutput(npu::utils::toFP32(reference), imgWidth, imgHeight,
                                                  static_cast<float>(confThresh), isTiny);

    bool result = checkBBoxOutputs(parsedOutput, parsedReference, imgWidth, imgHeight, static_cast<float>(boxTolerance),
                                   static_cast<float>(probTolerance));
    return result;
}

//
// Yolo V3 mode
//
bool testYoloV3(const TensorMap& outputs, const TensorMap& references, const TensorDescriptorMap& inputDescriptors,
                const LayoutMap& outputLayouts, size_t batch_size = 1) {
    if (batch_size != 1) {
        throw std::runtime_error(
                "The testcase 'yolo_v3' doesn't support any `override_model_batch_size` values besides 1 yet");
    }
    OPENVINO_ASSERT(inputDescriptors.size() == 1, "The YOLO v3 model accepts only a single input");
    OPENVINO_ASSERT(outputs.size() == 3, "The YOLO v3 model has three outputs");
    OPENVINO_ASSERT(outputs.size() == references.size(),
                    "Mismatch between the number of model outputs and the number of references");

    const TensorDescriptor& inputDescriptor = inputDescriptors.begin()->second;
    const auto imgWidth = inputDescriptor.dataShape.at(ov::layout::width_idx(inputDescriptor.layout));
    const auto imgHeight = inputDescriptor.dataShape.at(ov::layout::height_idx(inputDescriptor.layout));

    double confThresh = FLAGS_confidence_threshold;
    double probTolerance = FLAGS_prob_tolerance;
    double boxTolerance = FLAGS_box_tolerance;
    int classes = FLAGS_classes;
    int coords = FLAGS_coords;
    int num = FLAGS_num;
    std::vector<float> anchors = {10.0, 13.0, 16.0,  30.0,  33.0, 23.0,  30.0,  61.0,  62.0,
                                  45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};

    auto parsedOutput = utils::parseYoloV3Output(outputs, imgWidth, imgHeight, classes, coords, num, anchors,
                                                 static_cast<float>(confThresh), outputLayouts);
    auto parsedReference = utils::parseYoloV3Output(references, imgWidth, imgHeight, classes, coords, num, anchors,
                                                    static_cast<float>(confThresh), outputLayouts);

    bool result = checkBBoxOutputs(parsedOutput, parsedReference, imgWidth, imgHeight, static_cast<float>(boxTolerance),
                                   static_cast<float>(probTolerance));
    return result;
}

//
// Yolo V4 mode
// Ref link: https://docs.openvino.ai/latest/omz_models_model_yolo_v4_tiny_tf.html
//
bool testYoloV4(const TensorMap& outputs, const TensorMap& references, const TensorDescriptorMap& inputDescriptors,
                const LayoutMap& outputLayouts, size_t batch_size = 1) {
    if (batch_size != 1) {
        throw std::runtime_error(
                "The testcase 'yolo_v4' doesn't support any `override_model_batch_size` values besides 1 yet");
    }

    OPENVINO_ASSERT(inputDescriptors.size() == 1, "The YOLO v4 model accepts only a single input");
    OPENVINO_ASSERT(outputs.size() == 2, "The YOLO v4 model has two outputs");
    OPENVINO_ASSERT(outputs.size() == references.size(),
                    "Mismatch between the number of model outputs and the number of references");

    const TensorDescriptor& inputDescriptor = inputDescriptors.begin()->second;
    const auto imgWidth = inputDescriptor.dataShape.at(ov::layout::width_idx(inputDescriptor.layout));
    const auto imgHeight = inputDescriptor.dataShape.at(ov::layout::height_idx(inputDescriptor.layout));

    double confThresh = FLAGS_confidence_threshold;
    double probTolerance = FLAGS_prob_tolerance;
    double boxTolerance = FLAGS_box_tolerance;
    int classes = FLAGS_classes;
    int coords = FLAGS_coords;
    int num = FLAGS_num;
    std::vector<float> anchors = {10.0, 14.0, 23.0, 27.0, 37.0, 58.0, 81.0, 82.0, 135.0, 169.0, 344.0, 319.0};
    std::vector<std::vector<float>> anchor_mask{{3, 4, 5}, {1, 2, 3}};
    std::vector<float> masked_anchors{};
    for (auto& it : anchor_mask) {
        int index = 0;
        for (auto& anchorIndex : it) {
            if (index >= num)
                break;

            index++;
            masked_anchors.push_back(anchors[static_cast<size_t>(2 * anchorIndex)]);
            masked_anchors.push_back(anchors[static_cast<size_t>(2 * anchorIndex + 1)]);
        }
    }

    auto refOutput = utils::parseYoloV4Output(references, imgWidth, imgHeight, classes, coords, num, masked_anchors,
                                              static_cast<float>(confThresh), outputLayouts);
    auto actOutput = utils::parseYoloV4Output(outputs, imgWidth, imgHeight, classes, coords, num, masked_anchors,
                                              static_cast<float>(confThresh), outputLayouts);
    bool result = checkBBoxOutputs(actOutput, refOutput, imgWidth, imgHeight, static_cast<float>(boxTolerance),
                                   static_cast<float>(probTolerance));
    return result;
}

//
// MeanIoU mode
// Using sem_seg_classes, sem_seg_threshold flags and optionally sem_seg_ignore_label and dataset flags for validation
// e.g. '--mode mean_iou --sem_seg_classes 12 --sem_seg_threshold 0.98 --sem_seg_ignore_label 11 --dataset camVid12'
//
bool testMeanIoU(const TensorMap& outputs, const TensorMap& references, const LayoutMap& outputLayouts,
                 size_t batch_size = 1) {
    if (batch_size != 1) {
        throw std::runtime_error(
                "The testcase 'mean_iou' doesn't support any `override_model_batch_size` values besides 1 yet");
    }

    OPENVINO_ASSERT(outputs.size() == 1, "The metric accepts only a single output");
    OPENVINO_ASSERT(outputs.size() == references.size(),
                    "Mismatch between the number of model outputs and the number of references");
    OPENVINO_ASSERT(outputs.size() == outputLayouts.size(),
                    "Mismatch between the number of model outputs and their corresponding layout values");

    unsigned int classes = FLAGS_sem_seg_classes;
    auto semSegThreshold = static_cast<float>(FLAGS_sem_seg_threshold);

    std::vector<uint8_t> parsedReferences;
    std::vector<uint8_t> parsedOutputs;
    std::vector<std::pair<bool, float>> iou(classes, {false, 0.0f});

    utils::argMax_channels(references.begin()->second, parsedReferences, outputLayouts.begin()->second);
    utils::argMax_channels(outputs.begin()->second, parsedOutputs, outputLayouts.begin()->second);

    if (parsedReferences.size() != parsedOutputs.size()) {
        std::cout << "Reference size and output size are different" << std::endl;
        return false;
    }
    iou = utils::mean_IoU(parsedOutputs, parsedReferences, classes, FLAGS_sem_seg_ignore_label);

    return compare_mean_IoU(iou, semSegThreshold, classes);
}

static ov::Shape parseDataShape(const std::string& dataShapeStr) {
    std::vector<size_t> dataShape;
    std::istringstream ss(dataShapeStr);
    std::string token;
    while (std::getline(ss, token, ',')) {
        dataShape.push_back(std::stoul(token));
    }
    return ov::Shape(dataShape);
}

static int runSingleImageTest() {
    std::cout << "Run single image test" << std::endl;
    try {
        const std::unordered_set<std::string> allowedPrecision = {"U8", "I32", "I64", "FP16", "FP32"};
        if (!FLAGS_ip.empty()) {
            // input precision is U8, I32, I64, FP16 or FP32 only
            std::transform(FLAGS_ip.begin(), FLAGS_ip.end(), FLAGS_ip.begin(), ::toupper);
            if (allowedPrecision.count(FLAGS_ip) == 0)
                throw std::logic_error("Parameter -ip " + FLAGS_ip + " is not supported");
        }
        if (!FLAGS_op.empty()) {
            // output precision is U8, I32, I64, FP16 or FP32 only
            std::transform(FLAGS_op.begin(), FLAGS_op.end(), FLAGS_op.begin(), ::toupper);
            if (allowedPrecision.count(FLAGS_op) == 0)
                throw std::logic_error("Parameter -op " + FLAGS_op + " is not supported");
        }

        std::map<RegexPtr, ov::Layout> inUserLayouts = parseLayoutRegex(FLAGS_il);
        std::map<RegexPtr, ov::Layout> outUserLayouts = parseLayoutRegex(FLAGS_ol);
        std::map<RegexPtr, ov::Layout> inModelLayouts = parseLayoutRegex(FLAGS_iml);
        std::map<RegexPtr, ov::Layout> outModelLayouts = parseLayoutRegex(FLAGS_oml);

        std::vector<std::string> inputFilesPerCase;
        std::vector<std::vector<std::string>> inputFilesForOneInfer;

        inputFilesPerCase = splitStringList(FLAGS_input, ';');
        for (const auto& images : inputFilesPerCase) {
            inputFilesForOneInfer.push_back(splitStringList(images, ','));
        }

        std::vector<std::string> inputBinPrecisionStrPerCase;
        std::vector<std::vector<ov::element::Type>> inputBinPrecisionForOneInfer(inputFilesForOneInfer.size());
        if (FLAGS_img_as_bin) {
            for (std::size_t i = 0; i < inputFilesForOneInfer.size(); ++i) {
                inputBinPrecisionForOneInfer[i] =
                        std::vector<ov::element::Type>(inputFilesForOneInfer[i].size(), ov::element::undefined);
            }
            inputBinPrecisionStrPerCase = splitStringList(FLAGS_img_bin_precision, ';');
            std::size_t inferIdx = 0;
            for (const auto& precisions : inputBinPrecisionStrPerCase) {
                std::vector<std::string> inputBinPrecisionsStrThisInfer = splitStringList(precisions, ',');
                std::size_t precisionIdx = 0;
                for (const auto& precision : inputBinPrecisionsStrThisInfer) {
                    if (strEq(precision, "FP32")) {
                        inputBinPrecisionForOneInfer[inferIdx][precisionIdx] = ov::element::f32;
                    } else if (strEq(precision, "FP16")) {
                        inputBinPrecisionForOneInfer[inferIdx][precisionIdx] = ov::element::f16;
                    } else if (strEq(precision, "BF16")) {
                        inputBinPrecisionForOneInfer[inferIdx][precisionIdx] = ov::element::bf16;
                    } else if (strEq(precision, "I32")) {
                        inputBinPrecisionForOneInfer[inferIdx][precisionIdx] = ov::element::i32;
                    } else if (strEq(precision, "I64")) {
                        inputBinPrecisionForOneInfer[inferIdx][precisionIdx] = ov::element::i64;
                    } else if (strEq(precision, "U8")) {
                        inputBinPrecisionForOneInfer[inferIdx][precisionIdx] = ov::element::u8;
                    } else {
                        std::cout << "WARNING: Unhandled precision '" << precision
                                  << "'! Only FP32, FP16, I32, I64 and U8 can be currently converted to the network's"
                                  << "input tensor precision.";
                    }
                    ++precisionIdx;
                }
                ++inferIdx;
            }
        }

        if (FLAGS_network.empty()) {
            std::cout << "Not enough parameters. Check help." << std::endl;
            return EXIT_FAILURE;
        }

        ov::Core core;
        setupOVCore(core);

        ov::CompiledModel compiledModel;
        if (hasLoadableExt(FLAGS_network)) {
            std::cout << "Load network " << FLAGS_network << std::endl;

            auto model = core.read_model(FLAGS_network);
            nameIOTensors(model);

            auto inputs_info = std::const_pointer_cast<ov::Model>(model)->inputs();
            InputsInfo info_map;

            std::cout << "Performing reshape" << std::endl;
            reshape(std::move(inputs_info), info_map, model, FLAGS_shape,
                    FLAGS_override_model_batch_size, FLAGS_device);

            ov::preprocess::PrePostProcessor ppp(model);

            // Input precision
            const auto inputInfo = model->inputs();
            if (!FLAGS_ip.empty()) {
                ov::element::Type prc_in = ov::element::u8;
                if (FLAGS_ip == "FP16")
                    prc_in = ov::element::f16;
                else if (FLAGS_ip == "BF16")
                    prc_in = ov::element::bf16;
                else if (FLAGS_ip == "FP32")
                    prc_in = ov::element::f32;
                else if (FLAGS_ip == "I32")
                    prc_in = ov::element::i32;
                else if (FLAGS_ip == "I64")
                    prc_in = ov::element::i64;
                else
                    prc_in = ov::element::u8;

                for (size_t i = 0; i < inputInfo.size(); ++i) {
                    ppp.input(i).tensor().set_element_type(prc_in);
                }
            }

            // Input layout
            for (size_t i = 0; i < inputInfo.size(); ++i) {
                if (std::optional<ov::Layout> inUserLayout =
                            getRegexSubstitutionIfExist(inputInfo[i].get_any_name(), inUserLayouts);
                    inUserLayout.has_value()) {
                    ov::Layout inLayerModelLayout;
                    if (std::optional<ov::Layout> inModelLayout =
                                getRegexSubstitutionIfExist(inputInfo[i].get_any_name(), inModelLayouts);
                        inModelLayout.has_value()) {
                        inLayerModelLayout = inModelLayout.value();
                    } else {
                        const auto shape = inputInfo[i].get_partial_shape();
                        inLayerModelLayout = getLayoutByRank(shape.size());
                        std::cout << "WARNING: Configuring preprocessing. Since --iml option isn't set, input model "
                                     "layout for layer \""
                                  << inputInfo[i].get_any_name() << "\" is infered from shape: " << shape.to_string()
                                  << " rank (" << shape.size() << ") as " << inLayerModelLayout.to_string()
                                  << std::endl;
                    }
                    std::cout << "Set layouts for the input: \"" << inputInfo[i].get_any_name() << "\", model "
                              << inLayerModelLayout.to_string() << ", user " << inUserLayout.value().to_string()
                              << std::endl;
                    ppp.input(i).model().set_layout(inLayerModelLayout);
                    ppp.input(i).tensor().set_layout(inUserLayout.value());
                }
            }

            // Input mean and scale if exist
            if (!FLAGS_mean_values.empty() || !FLAGS_scale_values.empty()) {
                auto means = parseMeanOrScale(FLAGS_mean_values, inputInfo);
                auto scales = parseMeanOrScale(FLAGS_scale_values, inputInfo);
                for (size_t i = 0; i < inputInfo.size(); ++i) {
                    if (!means[i].empty()) {
                        ppp.input(i).preprocess().convert_element_type(ov::element::f32).mean(means[i]);
                    }
                    if (!scales[i].empty()) {
                        ppp.input(i).preprocess().convert_element_type(ov::element::f32).scale(scales[i]);
                    }
                }
            }

            // Output precision
            const auto outputInfo = model->outputs();
            if (!FLAGS_op.empty()) {
                ov::element::Type prc_out = ov::element::u8;
                if (FLAGS_op == "FP16")
                    prc_out = ov::element::f16;
                else if (FLAGS_op == "FP32")
                    prc_out = ov::element::f32;
                else if (FLAGS_op == "I32")
                    prc_out = ov::element::i32;
                else if (FLAGS_op == "I64")
                    prc_out = ov::element::i64;
                else
                    prc_out = ov::element::u8;

                for (size_t i = 0; i < outputInfo.size(); ++i) {
                    ppp.output(i).tensor().set_element_type(prc_out);
                }
            }

            // Output layout
            for (size_t i = 0; i < outputInfo.size(); ++i) {
                if (std::optional<ov::Layout> outUserLayout =
                            getRegexSubstitutionIfExist(outputInfo[i].get_any_name(), outUserLayouts);
                    outUserLayout.has_value()) {
                    ov::Layout outLayerModelLayout;
                    if (std::optional<ov::Layout> outModelLayout =
                                getRegexSubstitutionIfExist(outputInfo[i].get_any_name(), outModelLayouts);
                        outModelLayout.has_value()) {
                        outLayerModelLayout = outModelLayout.value();
                    } else {
                        const auto shape = outputInfo[i].get_partial_shape();
                        outLayerModelLayout = getLayoutByRank(shape.size());
                        std::cout << "WARNING: Configuring preprocessing. Since --oml option isn't set, output model "
                                     "layout for layer \""
                                  << outputInfo[i].get_any_name() << "\" is infered from shape: " << shape.to_shape()
                                  << " rank (" << shape.size() << ") as " << outLayerModelLayout.to_string()
                                  << std::endl;
                    }
                    std::cout << "Set layouts for the output: \"" << outputInfo[i].get_any_name() << "\", model "
                              << outLayerModelLayout.to_string() << ", user " << outUserLayout.value().to_string()
                              << std::endl;
                    ppp.output(i).model().set_layout(outLayerModelLayout);
                    ppp.output(i).tensor().set_layout(outUserLayout.value());
                }
            }

            std::cout << "Compile model" << std::endl;
            compiledModel = core.compile_model(ppp.build(), FLAGS_device);
        } else {
            std::cout << "Import network " << FLAGS_network << std::endl;

            if (!FLAGS_mean_values.empty() || !FLAGS_scale_values.empty()) {
                throw std::runtime_error("--mean_values and --scale_values aren't supported for "
                                         "compiled model.\n The values can be set via "
                                         "model_optimizer while generating xml\n");
            }

            std::ifstream file(FLAGS_network, std::ios_base::in | std::ios_base::binary);
            OPENVINO_ASSERT(file.is_open(), "Can't open file ", FLAGS_network, " for read");
            compiledModel = core.import_model(file, FLAGS_device);
        }

        // store compiled model, if required
        if (!FLAGS_compiled_blob.empty()) {
            std::ofstream outputFile{FLAGS_compiled_blob, std::ios::out | std::ios::binary};
            if (!outputFile.is_open()) {
                std::cerr << "Output file \"" << FLAGS_compiled_blob << "\" can't be opened for writing" << std::endl;
                return EXIT_FAILURE;
            } else {
                compiledModel.export_model(outputFile);
            }
        }

        auto inferRequest = compiledModel.create_infer_request();

        std::string netFileName;
        {
            auto startPos = FLAGS_network.rfind('/');
            if (startPos == std::string::npos) {
                startPos = FLAGS_network.rfind('\\');
                if (startPos == std::string::npos) {
                    startPos = 0;
                }
            }

            auto endPos = FLAGS_network.rfind('.');
            if (endPos == std::string::npos) {
                endPos = FLAGS_network.size();
            }

            OPENVINO_ASSERT(endPos > startPos);
            netFileName = cleanName(FLAGS_network.substr(startPos, endPos - startPos));
        }

        for (size_t numberOfTestCase = 0; numberOfTestCase < inputFilesPerCase.size(); ++numberOfTestCase) {
            const auto inputsInfo = compiledModel.inputs();
            const auto outputsInfo = compiledModel.outputs();
            std::vector<std::string> inputFiles = inputFilesForOneInfer[numberOfTestCase];
            OPENVINO_ASSERT(inputFiles.size() == inputsInfo.size(), "Number of input files ", inputFiles.size(),
                            " doesn't match network configuration ", inputsInfo.size());

            TensorMap inTensors;
            size_t inputInd = 0;
            std::vector<std::string> dumpedInputsPaths;
            TensorDescriptorMap inputDescriptors;  // Several metrics require the input metadata

            // Load the input data
            for (const auto& inputInfo : inputsInfo) {
                const auto& shape = inputInfo.get_partial_shape();
                const auto dataShape = shape.is_static() ? shape.get_shape() : parseDataShape(FLAGS_data_shape);
                const ov::element::Type& precision = inputInfo.get_element_type();

                // Determine the input layout
                ov::Layout inputLayout;

                if (std::optional<ov::Layout> inUserLayout =
                            getRegexSubstitutionIfExist(inputInfo.get_any_name(), inUserLayouts);
                    inUserLayout.has_value()) {
                    inputLayout = inUserLayout.value();
                } else if (std::optional<ov::Layout> inModelLayout =
                                   getRegexSubstitutionIfExist(inputInfo.get_any_name(), inModelLayouts);
                           inModelLayout.has_value()) {
                    inputLayout = inModelLayout.value();
                } else {
                    inputLayout = getLayoutByRank(shape.size());
                    std::cout << "WARNING: Loading input data. Since --iml option isn't set, input model layout for "
                                 "layer \""
                              << inputInfo.get_any_name() << "\" is infered from shape: " << shape.to_shape()
                              << " rank (" << shape.size() << ") as " << inputLayout.to_string() << std::endl;
                }

                inputDescriptors.emplace(inputInfo.get_any_name(), TensorDescriptor{precision, shape,
                                                                                    dataShape, inputLayout});

                std::cout << "Load input #" << inputInd << " from " << inputFiles[inputInd] << " as " << precision
                          << " " << inputLayout.to_string() << " " << shape << std::endl;

                const ov::Tensor tensor =
                        !FLAGS_img_as_bin
                                ? loadInput(precision, dataShape, inputLayout, inputFiles[inputInd], FLAGS_color_format)
                                : loadInput(precision, dataShape, inputLayout, inputFiles[inputInd], FLAGS_color_format,
                                            inputBinPrecisionForOneInfer[numberOfTestCase][inputInd]);
                std::ostringstream ostr;
                ostr << netFileName << "_input_" << inputInd << "_case_" << numberOfTestCase << ".blob";
                const auto blobFileName = ostr.str();

                std::cout << "Dump input #" << inputInd << "_case_" << numberOfTestCase << " to " << blobFileName
                          << std::endl;
                dumpTensor(tensor, blobFileName);

                ++inputInd;

                dumpedInputsPaths.push_back(blobFileName);

                inTensors.emplace(inputInfo.get_any_name(), std::move(tensor));
            }

            std::cout << "Run inference on " << FLAGS_device << std::endl;

            const auto startTime = Time::now();
            const auto outInference = runInfer(inferRequest, compiledModel, inTensors, dumpedInputsPaths);
            const auto endTime = Time::now();

            const TensorMap& outputTensors = outInference.first;

            printPerformanceCountsAndLatency(numberOfTestCase, outInference.second, endTime - startTime);

            if (FLAGS_run_test) {
                TensorMap referenceTensors;
                size_t outputInd = 0;
                LayoutMap outputLayouts;  // Several metrics may require this

                // Load the reference data
                for (const auto& out : compiledModel.outputs()) {
                    const auto& tensorName = out.get_any_name();
                    const auto& tensor = outputTensors.at(tensorName);
                    const ov::element::Type& precision = tensor.get_element_type();
                    const ov::Shape& shape = tensor.get_shape();

                    std::ostringstream ostr;
                    ostr << netFileName << "_ref_out_" << outputInd << "_case_" << numberOfTestCase << ".blob";
                    const auto blobFileName = ostr.str();

                    std::filesystem::path fullPath = FLAGS_ref_dir;
                    fullPath /= blobFileName;
                    const auto blobFileFullName = fullPath.string();

                    std::cout << "Load reference output #" << outputInd << " from " << blobFileFullName << " as "
                              << precision << std::endl;

                    const ov::Tensor referenceTensor = loadTensor(precision, shape, blobFileFullName);
                    referenceTensors.emplace(tensorName, referenceTensor);

                    // Determine the output layout
                    ov::Layout outputLayout;

                    if (std::optional<ov::Layout> outUserLayout =
                                getRegexSubstitutionIfExist(tensorName, outUserLayouts);
                        outUserLayout.has_value()) {
                        outputLayout = outUserLayout.value();
                    } else if (std::optional<ov::Layout> outModelLayout =
                                       getRegexSubstitutionIfExist(tensorName, outModelLayouts);
                               outModelLayout.has_value()) {
                        outputLayout = outModelLayout.value();
                    } else {
                        outputLayout = getLayoutByRank(shape.size());
                        std::cout << "WARNING: Since --oml option isn't set, output model layout for layer \""
                                  << tensorName << "\" is infered from shape: " << toString(shape) << " rank ("
                                  << shape.size() << ") as " << outputLayout.to_string() << std::endl;
                    }

                    outputLayouts.emplace(tensorName, outputLayout);

                    ++outputInd;
                }

                outputInd = 0;

                // Dump the outputs obtained upon prediction
                for (const auto& out : compiledModel.outputs()) {
                    const auto& tensor = outputTensors.at(out.get_any_name());
                    std::ostringstream ostr;
                    ostr << netFileName << "_kmb_out_" << outputInd << "_case_" << numberOfTestCase << ".blob";
                    const auto blobFileName = ostr.str();

                    std::cout << "Dump device output #" << outputInd << "_case_" << numberOfTestCase << " to "
                              << blobFileName << std::endl;

                    dumpTensor(tensor, blobFileName);
                    ++outputInd;
                }

                // Compare the outputs with their references using the chosen metric
                if (strEq(FLAGS_mode, "classification")) {
                    if (testClassification(outputTensors, referenceTensors, FLAGS_override_model_batch_size)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "raw")) {
                    if (testRAW(outputTensors, referenceTensors, FLAGS_override_model_batch_size)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "cosim")) {
                    if (testCoSim(outputTensors, referenceTensors, FLAGS_override_model_batch_size)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "rrmse")) {
                    if (testRRMSE(outputTensors, referenceTensors, FLAGS_override_model_batch_size)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "nrmse")) {
                    if (testNRMSE(outputTensors, referenceTensors, FLAGS_override_model_batch_size)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "ssd")) {
                    if (testSSDDetection(outputTensors, referenceTensors, inputDescriptors,
                                         FLAGS_override_model_batch_size)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "yolo_v2")) {
                    if (testYoloV2(outputTensors, referenceTensors, inputDescriptors,
                                   FLAGS_override_model_batch_size)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "yolo_v3")) {
                    if (testYoloV3(outputTensors, referenceTensors, inputDescriptors, outputLayouts,
                                   FLAGS_override_model_batch_size)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "yolo_v4")) {
                    if (testYoloV4(outputTensors, referenceTensors, inputDescriptors, outputLayouts,
                                   FLAGS_override_model_batch_size)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "psnr")) {
                    const auto& [firstOutputName, firstOutput] = *outputTensors.begin();
                    const ov::Shape& shape = firstOutput.get_shape();
                    const ov::Layout& outputLayout = outputLayouts.at(firstOutputName);
                    const size_t dstHeight = shape[ov::layout::height_idx(outputLayout)];
                    const size_t dstWidth = shape[ov::layout::width_idx(outputLayout)];

                    if (testPSNR(outputTensors, referenceTensors, static_cast<int>(dstHeight),
                                 static_cast<int>(dstWidth), FLAGS_override_model_batch_size)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "mean_iou")) {
                    if (testMeanIoU(outputTensors, referenceTensors, outputLayouts, FLAGS_override_model_batch_size)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else {
                    std::cout << "Unknown mode " << FLAGS_mode << std::endl;
                    return EXIT_FAILURE;
                }
            } else {
                size_t outputInd = 0;
                for (const auto& out : compiledModel.outputs()) {
                    const auto& tensor = outputTensors.at(out.get_any_name());
                    std::ostringstream ostr;
                    ostr << netFileName << "_ref_out_" << outputInd << "_case_" << numberOfTestCase << ".blob";
                    const auto blobFileName = ostr.str();

                    std::cout << "Dump reference output #" << outputInd << " to " << blobFileName << std::endl;
                    dumpTensor(tensor, blobFileName);

                    ++outputInd;
                }
            }
        }
    }  // try
    catch (const std::exception& ex) {
        std::cerr << "exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

//
// main
//

int main(int argc, char* argv[]) {
    parseCommandLine(argc, argv);

    return runSingleImageTest();
}
