// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
namespace GNAPluginNS {
/**
 * @brief policy agregates various settings that cannot be tweak using configuration options right now,
 * and essential to keep test coverage for options both in on and off cases
 */
class Policy {
 public:
    /**
    * @brief for scaleshift substitution, weight tiling simplify final graph but have extra weights overhead
    * if not defined scaleshift broadcast will result in creating multiple diagonal layers instead of weight tiling
    */
    enum class ScaleShift {
        WEIGHTS_TILING,
        /**
         * GNA has limited amount of batch so even existed topologies cannot be substituted with only batching,
         * this option combines batch and weights tiling
         */
        BATCH_AND_WEIGHTS_TILING,
        DIAGLAYER_TILING
    } ScaleShiftPolicy = ScaleShift::WEIGHTS_TILING;

    /**
     * Policy on whether to substitute permute layers or not
     */
    enum class Permute {
        DISABLED,
        AUTO_PERMUTE
    } PermutePolicy = Permute::DISABLED;

    enum class ConcatAlignment {
        DISABLED,
        DISABLED_FOR_FP32,
        ENABLED
    } ConcatAlignmentPolicy = ConcatAlignment::ENABLED;
};

inline std::ostream& operator<<(std::ostream& os, Policy::ScaleShift policy) {
    switch (policy) {
        case Policy::ScaleShift::WEIGHTS_TILING   : os << "WEIGHTS_TILING";    break;
        case Policy::ScaleShift::BATCH_AND_WEIGHTS_TILING: os << "BATCH_AND_WEIGHTS_TILING"; break;
        case Policy::ScaleShift::DIAGLAYER_TILING : os << "DIAGLAYER_TILING";  break;
        default    : os.setstate(std::ios_base::failbit);
    }
    return os;
}

}  // namespace GNAPluginNS
