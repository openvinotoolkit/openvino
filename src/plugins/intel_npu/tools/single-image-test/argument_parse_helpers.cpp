// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "argument_parse_helpers.hpp"
#include <openvino/openvino.hpp>
#include <iostream>
#include <sstream>
#include <string>

// Non-template globals declared extern in the header
CaselessEq<std::string> strEq;
std::vector<std::string> camVid12 = {"Sky",        "Building", "Pole", "Road",       "Pavement",  "Tree",
                                     "SignSymbol", "Fence",    "Car",  "Pedestrian", "Bicyclist", "Unlabeled"};

//
// Command line flag definitions
//

DEFINE_string(network, "", "Network file (either XML or pre-compiled blob)");
DEFINE_string(input, "", "Input file(s)");
DEFINE_string(compiled_blob, "", "Output compiled network file (compiled result blob)");
DEFINE_uint32(override_model_batch_size, 1, "Enforce a model to be compiled for batch size");
DEFINE_string(device, "", "Device to use");
DEFINE_string(config, "", "Path to the configuration file (optional)");
DEFINE_string(ip, "", "Input precision (default: U8, available: FP32, FP16, I32, I64, U8, U16,"
                      " I16, U4, I4, U2, BF8(F8E5M2), HF8(F8E4M3))");
DEFINE_string(op, "", "Output precision (default: FP32, available: FP32, FP16, I32, I64, U8, U16,"
                      " I16, U4, I4, U2, BF8(F8E5M2), HF8(F8E4M3))");
DEFINE_string(il, "",
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
DEFINE_string(shape, "",
        "Optional. Set shape for model input. For example, \"input1[1,3,224,224],input2[1,4]\" or \"[1,3,224,224]\""
        " in case of one input size. This parameter affects model input shape and can be dynamic."
        " For dynamic dimensions use symbol `?` or '-1'. Ex. [?,3,?,?]."
        " For bounded dimensions specify range 'min..max'. Ex. [1..10,3,?,?].");
DEFINE_string(data_shape, "",
        "Required for models with dynamic shapes. Set shape for input blobs. Only one shape can be set."
        "In case of one input size: \"[1,3,224,224]\"");
DEFINE_string(skip_output_layers, "", "Skip output layers from the network."
        " Accept ';' separated list of output layers");
DEFINE_bool(clamp_u8_outputs, false, "Apply clamping when converting FP to U8");
DEFINE_string(mean_values, "",
        "Optional. Mean values to be used for the input image per channel. "
        "Values to be provided in the [channel1,channel2,channel3] format. "
        "Can be defined for desired input of the model, for example: \"--mean_values "
        "data[255,255,255],info[255,255,255]\". The exact meaning and order of channels depend on how the original "
        "model was trained. Applying the values affects performance and may cause type conversion");
DEFINE_string(scale_values, "",
        "Optional. Scale values to be used for the input image per channel. "
        "Values are provided in the [channel1,channel2,channel3] format. "
        "Can be defined for desired input of the model, for example: \"--scale_values "
        "data[255,255,255],info[255,255,255]\". "
        "The exact meaning and order of channels depend on how the original model was trained. If both --mean_values "
        "and --scale_values are specified, the mean is subtracted first and then scale is applied regardless of the "
        "order of options in command line. Applying the values affects performance and may cause type conversion");
DEFINE_string(img_bin_precision, "", "Specify the precision of the binary input files. Eg: 'FP32,FP16,I32,I64,U8'");
DEFINE_bool(run_test, false, "Run the test (compare current results with previously dumped)");
DEFINE_string(ref_dir, "",
        "A directory with reference blobs to compare with in run_test mode. Leave it empty to use the current folder.");
DEFINE_string(ref_results, "",
        "String of reference result file(s) to be used during run_test mode. "
        "For the same test case, the files should be separated by comma (,) (example: one case multiple output). "
        "For different test cases, it should be separated by semicolon (;). "
        "If ref_dir is provided, the reference files should be relative to the ref_dir. "
        "Else, if ref_dir is not provided, the reference files should be absolute paths.");
DEFINE_string(mode, "", "Comparison mode to use");
DEFINE_uint32(top_k, 1, "Top K parameter for 'classification' mode");
DEFINE_string(prob_tolerance, std::to_string(metric_defaults::prob_tolerance),
        "Probability tolerance for 'classification/ssd/yolo' mode. "
        "Can be a single value or per-layer: 'layer1:0.01;layer2:0.02'");
DEFINE_string(raw_tolerance, std::to_string(metric_defaults::raw_tolerance),
        "Tolerance for 'raw' mode (absolute diff). Can be a single value or per-layer: 'layer1:0.01;layer2:0.02'");
DEFINE_string(cosim_threshold, std::to_string(metric_defaults::cosim_threshold),
        "Threshold for 'cosim' mode. Can be a single value or per-layer: 'layer1:0.95;layer2:0.90'");
DEFINE_string(rrmse_loss_threshold, std::to_string(metric_defaults::rrmse_loss_threshold),
        "Threshold for 'rrmse' mode. Can be a single value or per-layer: 'layer1:0.1;layer2:0.2'");
DEFINE_string(nrmse_loss_threshold, std::to_string(metric_defaults::nrmse_loss_threshold),
        "Threshold for 'nrmse' mode. Can be a single value or per-layer: 'logits:0.03;pred_boxes:0.05'");
DEFINE_string(l2norm_threshold, std::to_string(metric_defaults::l2norm_threshold),
        "Threshold for 'l2norm' mode. Can be a single value or per-layer: 'layer1:1.0;layer2:2.0'");
DEFINE_string(overlap_threshold, std::to_string(metric_defaults::overlap_threshold),
        "IoU threshold for 'map' mode (detection matching). " \
        "Can be a single value or per-layer: 'layer1:0.5;layer2:0.6'");
DEFINE_string(map_threshold, std::to_string(metric_defaults::map_threshold),
        "mAP score threshold for 'map' mode validation. Can be a single value or per-layer: 'layer1:0.5;layer2:0.6'");
DEFINE_string(confidence_threshold, std::to_string(metric_defaults::confidence_threshold),
        "Confidence threshold for Detection mode. Can be a single value or per-layer: 'layer1:0.5;layer2:0.3'");
DEFINE_string(box_tolerance, std::to_string(metric_defaults::box_tolerance),
        "Box tolerance for 'detection' mode. Can be a single value or per-layer: 'layer1:0.01;layer2:0.02'");
DEFINE_bool(apply_soft_max, false, "Apply SoftMax for 'nrmse' mode");
DEFINE_string(psnr_reference, std::to_string(metric_defaults::psnr_reference),
        "PSNR reference value in dB. Can be a single value or per-layer: 'layer1:30.0;layer2:35.0'");
DEFINE_string(psnr_tolerance, std::to_string(metric_defaults::psnr_tolerance),
        "Tolerance for 'psnr' mode. Can be a single value or per-layer: 'layer1:0.01;layer2:0.02'");
DEFINE_string(log_level, "", "IE logger level (optional)");
DEFINE_string(color_format, "BGR", "Color format for input: RGB or BGR");
DEFINE_uint32(scale_border, 4, "Scale border");
DEFINE_bool(normalized_image, false, "Images in [0, 1] range or not");

// for Yolo
DEFINE_bool(is_tiny_yolo, false, "Is it Tiny Yolo or not (true or false)?");
DEFINE_int32(classes, 80, "Number of classes for Yolo V3");
DEFINE_int32(coords, 4, "Number of coordinates for Yolo V3");
DEFINE_int32(num, 3, "Number of scales for Yolo V3");

// for Semantic Segmentation
DEFINE_bool(skip_arg_max, false, "Skip ArgMax post processing step");
DEFINE_uint32(sem_seg_classes, 12, "Number of classes for semantic segmentation");
DEFINE_string(sem_seg_threshold, "0.98",
        "Threshold for 'semantic segmentation' mode. Can be a single value or per-layer: 'layer1:0.98;layer2:0.95'");
DEFINE_uint32(sem_seg_ignore_label, std::numeric_limits<uint32_t>::max(), "The number of the label to be ignored");
DEFINE_string(dataset, "NONE",
        "The dataset used to train the model. Useful for instances such as semantic segmentation to visualize "
        "the accuracy per-class");


void utils::parseCommandLine(int argc, char* argv[]) {
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
    std::cout << "    Skip checking output layers:              " << FLAGS_skip_output_layers << std::endl;
    std::cout << "    Clamp U8 outputs:                         " << FLAGS_clamp_u8_outputs << std::endl;
    if (FLAGS_run_test) {
        std::cout << "    Reference files directory:                "
                  << (FLAGS_ref_dir.empty() && FLAGS_ref_results.empty() ? "Current directory" : FLAGS_ref_dir)
                  << std::endl;
        std::cout << "    Reference file(s):                        " << FLAGS_ref_results << std::endl;
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
        } else if (strEq(FLAGS_mode, "map")) {
            std::cout << "    Overlap Threshold: " << FLAGS_overlap_threshold << std::endl;
            std::cout << "    mAP Threshold:     " << FLAGS_map_threshold << std::endl;
        } else if (strEq(FLAGS_mode, "nrmse")) {
            std::cout << "    Threshold:        " << FLAGS_nrmse_loss_threshold << std::endl;
        } else if (strEq(FLAGS_mode, "l2norm")) {
            std::cout << "    Threshold:        " << FLAGS_l2norm_threshold << std::endl;
        }
    }
    std::cout << "    Log level:                        " << FLAGS_log_level << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Parse a string of per-layer values into a map.
 * @param str Input string in format "layer1:value1;layer2:value2" or a single default value
 * @param defaultValue Default value to use if the string is a single number
 * @return Map of layer name to value
 * @example parsePerLayerValues("logits:0.03;pred_boxes:0.05", 1.0)
 *          returns {"logits": 0.03, "pred_boxes": 0.05}
 * @example parsePerLayerValues("0.01", 1.0) returns {"*": 0.01}
 */
utils::PerLayerValueMap utils::parsePerLayerValues(const std::string& str, double defaultValue) {
    PerLayerValueMap result;

    // Always store the default as the wildcard fallback so getValueForLayer
    // never needs a separate default parameter.
    result["*"] = defaultValue;

    if (str.empty()) {
        return result;
    }

    // Try to parse as a single number first
    try {
        double value = std::stod(str);
        result["*"] = value;
        return result;
    } catch (...) {
        // Not a single number, parse as key:value pairs
    }

    // Parse "layer1:value1;layer2:value2" format
    std::istringstream stream(str);
    std::string pair;

    while (std::getline(stream, pair, ';')) {
        size_t colonPos = pair.find(':');
        if (colonPos != std::string::npos) {
            std::string layerName = pair.substr(0, colonPos);
            std::string valueStr = pair.substr(colonPos + 1);

            // Trim whitespace
            layerName.erase(0, layerName.find_first_not_of(" \t"));
            layerName.erase(layerName.find_last_not_of(" \t") + 1);
            valueStr.erase(0, valueStr.find_first_not_of(" \t"));
            valueStr.erase(valueStr.find_last_not_of(" \t") + 1);

            try {
                double value = std::stod(valueStr);
                result[layerName] = value;
            } catch (const std::exception&) {
                std::cerr << "Warning: Failed to parse value '" << valueStr << "' for layer '" << layerName << "'" << std::endl;
            }
        }
    }

    return result;
}

/**
 * @brief Get the threshold value for a specific layer.
 * @param valueMap Map of layer name to value (always contains "*" fallback when
 *        created via parsePerLayerValues)
 * @param layerName Name of the layer
 * @return The threshold value for the layer
 */
double utils::getValueForLayer(const PerLayerValueMap& valueMap, const std::string& layerName) {
    // First try exact match
    auto it = valueMap.find(layerName);
    if (it != valueMap.end()) {
        return it->second;
    }

    // Fall back to wildcard (always present when map was created with parsePerLayerValues)
    it = valueMap.find("*");
    if (it != valueMap.end()) {
        return it->second;
    }

    // Should never be reached for properly initialised maps.
    return 0.0;
}
