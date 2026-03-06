//
// Copyright (C) 2018-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <map>
#include <string>
#include <vector>

#include <gflags/gflags.h>

// Default values for metric thresholds, referenced by flag definitions in argument_parse_helpers.cpp.
namespace metric_defaults {
    constexpr double prob_tolerance       = 1e-4;
    constexpr double raw_tolerance        = 1e-4;
    constexpr double cosim_threshold      = 0.90;
    constexpr double rrmse_loss_threshold = std::numeric_limits<double>::max();
    constexpr double nrmse_loss_threshold = 0.02;
    constexpr double l2norm_threshold     = 1.0;
    constexpr double overlap_threshold    = 0.50;
    constexpr double map_threshold        = 0.50;
    constexpr double confidence_threshold = 1e-4;
    constexpr double box_tolerance        = 1e-4;
    constexpr double psnr_reference       = 30.0;
    constexpr double psnr_tolerance       = 1e-4;
    constexpr double sem_seg_threshold    = 0.98;
} // namespace metric_defaults

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

extern CaselessEq<std::string> strEq;
extern std::vector<std::string> camVid12;

//
// Command line flags (defined in argument_parse_helpers.cpp)
//

DECLARE_string(network);
DECLARE_string(input);
DECLARE_string(compiled_blob);
DECLARE_uint32(override_model_batch_size);
DECLARE_string(device);
DECLARE_string(config);
DECLARE_string(ip);
DECLARE_string(op);
DECLARE_string(il);
DECLARE_string(ol);
DECLARE_string(iml);
DECLARE_string(oml);
DECLARE_bool(img_as_bin);
DECLARE_bool(pc);
DECLARE_string(shape);
DECLARE_string(data_shape);
DECLARE_string(skip_output_layers);
DECLARE_bool(clamp_u8_outputs);
DECLARE_string(mean_values);
DECLARE_string(scale_values);
DECLARE_string(img_bin_precision);
DECLARE_bool(run_test);
DECLARE_string(ref_dir);
DECLARE_string(ref_results);
DECLARE_string(mode);
DECLARE_uint32(top_k);
DECLARE_string(prob_tolerance);
DECLARE_string(raw_tolerance);
DECLARE_string(cosim_threshold);
DECLARE_string(rrmse_loss_threshold);
DECLARE_string(nrmse_loss_threshold);
DECLARE_string(l2norm_threshold);
DECLARE_string(overlap_threshold);
DECLARE_string(map_threshold);
DECLARE_string(confidence_threshold);
DECLARE_string(box_tolerance);
DECLARE_bool(apply_soft_max);
DECLARE_string(psnr_reference);
DECLARE_string(psnr_tolerance);
DECLARE_string(log_level);
DECLARE_string(color_format);
DECLARE_uint32(scale_border);
DECLARE_bool(normalized_image);
DECLARE_bool(is_tiny_yolo);
DECLARE_int32(classes);
DECLARE_int32(coords);
DECLARE_int32(num);
DECLARE_bool(skip_arg_max);
DECLARE_uint32(sem_seg_classes);
DECLARE_string(sem_seg_threshold);
DECLARE_uint32(sem_seg_ignore_label);
DECLARE_string(dataset);

namespace utils {

void parseCommandLine(int argc, char* argv[]);

using PerLayerValueMap = std::map<std::string, double>;

PerLayerValueMap parsePerLayerValues(const std::string& str, double defaultValue);
double getValueForLayer(const PerLayerValueMap& valueMap, const std::string& layerName);

}  // namespace utils
