// Copyright (C) 2018-2021 Intel Corporation
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

    enum class FlattenTrivialConcatConversion {
        DISABLED,
        ENABLED
    } ConcatConversionPolicy = FlattenTrivialConcatConversion::ENABLED;

    enum class ConcatAlignment {
        DISABLED,
        DISABLED_FOR_FP32,
        ENABLED,
        FAST
    } ConcatAlignmentPolicy = ConcatAlignment::FAST;

    /**
    * Policy to support --disable_nhwc_to_nchw option in MO
    */
    enum class NHWCToNCHW {
        DISABLED,
        REMOVE_LAST,
        REMOVE_ALL
    } NHWCToNCHWPolicy = NHWCToNCHW::REMOVE_ALL;

 /**
 * @brief trim of gna diagonal affine layer maximum elements number
 */
    class GNAAffineDiagonal {
    public:
        enum : uint32_t {
            UNLIMIT,
            // gna limit this to be OxFFFF
            LIMITED_TO_DEFAULT_GNA2_65536 = 65536 - 64
        };
        uint32_t limitedTo = LIMITED_TO_DEFAULT_GNA2_65536;
    } GNAAffineDiagonalPolicy;

    bool cnn2dInputPaddingSupported = false;
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

inline std::ostream& operator<<(std::ostream& os, Policy::ConcatAlignment policy) {
    switch (policy) {
        case Policy::ConcatAlignment::DISABLED   : os << "DISABLED";    break;
        case Policy::ConcatAlignment::DISABLED_FOR_FP32   : os << "DISABLED_FOR_FP32";    break;
        case Policy::ConcatAlignment::ENABLED   : os << "ENABLED";    break;
        case Policy::ConcatAlignment::FAST   : os << "FAST";    break;
        default    : os.setstate(std::ios_base::failbit);
    }
    return os;
}


}  // namespace GNAPluginNS
