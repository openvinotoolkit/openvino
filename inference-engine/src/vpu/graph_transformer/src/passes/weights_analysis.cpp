// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <cmath>

#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <string>
#include <algorithm>
#include <limits>
#include <memory>
#include <list>
#include <set>

#include <precision_utils.h>

#include <vpu/utils/numeric.hpp>

#include <details/caseless.hpp>

namespace vpu {

namespace {

const short largestExp   = 15;
const short smallestExp  = -15;
const short exponentBias = 15;

short calculateExponent(short src) {
    short exponent;
    const short numberBitsFP16Mantiss = 10;
    const short numberBitsFP16Exponent = 5;
    src = src >> numberBitsFP16Mantiss;

    exponent = (src & ((1 << numberBitsFP16Exponent)-1));
    exponent -= exponentBias;
    return exponent;
}

std::vector<short> calculateExponents(const fp16_t* srcPtr, int count) {
    std::vector<short> exponents(count);

    for (int i = 0; i < count; ++i) {
        exponents[i] = calculateExponent(srcPtr[i]);
    }

    return exponents;
}

short getModaValue(const std::vector<short>& exponents) {
    const int countExps = 32;
    std::vector<int> count(countExps);
    for (int i = 0; i < countExps; i++) {
        count[i] = 0;
    }

    for (int i = 0; i < exponents.size(); i++) {
        count[exponents[i] + exponentBias]++;
    }
    int medianIndex = 0;
    int medianValue = 0;

    for (int i = 0; i < countExps; i++) {
        if (count[i] > medianValue) {
            medianValue = count[i];
            medianIndex = i;
        }
    }
    return medianIndex - exponentBias;
}

int getMeanValue(const std::vector<short>& exponents) {
    double sum = 0;
    int realSize = 0;
    for (int i = 0; i < exponents.size(); i++) {
        if (exponents[i] != smallestExp) {
            sum += exponents[i];
            realSize++;
        }
    }

    if (realSize == 0) {
        return -15;
    } else {
        return sum / realSize;
    }
}

bool isScalable(const Stage& stage) {
    if (stage->type() != StageType::StubConv &&
        stage->type() != StageType::StubFullyConnected &&
        stage->type() != StageType::StubDeconv) {
        return false;
    }

    auto tryHW = stage->attrs().getOrDefault<bool>("tryHW", false);
    if (!tryHW) {
        return false;
    }

    return true;
}

class SingleScalePassImpl final : public Pass {
public:
    void run(const Model::Ptr& model) override;
};

void SingleScalePassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(estimateSingleNetworkScale);

    if (model->attrs().get<int>("numInputs") > 1) {
        return;
    }

    std::vector<short> modaExponents;
    std::vector<int> meanValueExponents;

    modaExponents.reserve(model->numStages());
    meanValueExponents.reserve(model->numStages());

    long long int bigAcc = 0.0;
    int realSize = 0;
    const int thresholdExp = -5;

    for (const auto& stage : model->getStages()) {
        if (!isScalable(stage)) {
            continue;
        }

        auto weights = stage->input(1);

        auto content = weights->content();
        IE_ASSERT(content != nullptr);

        auto weightsVals = content->get<fp16_t>();
        IE_ASSERT(weightsVals != nullptr);

        auto exponents =  calculateExponents(weightsVals, weights->desc().totalDimSize());

        modaExponents.emplace_back(getModaValue(exponents));
        meanValueExponents.emplace_back(getMeanValue(exponents));

        for (int i = 0; i <  exponents.size(); i++) {
            if (exponents[i] != smallestExp) {
                bigAcc += exponents[i];
                realSize++;
            }
        }
    }

    if (!meanValueExponents.empty()) {
        if (meanValueExponents[0] < thresholdExp) {
            if (realSize != 0) {
                model->attrs().set<int>("inputShift", (-1) * bigAcc / realSize);
            }
        }
    }
}

bool checkGrowingOutput(const Model::Ptr& model) {
    bool removeScale = true;

    for (const auto& stage : model->getStages()) {
        auto inputScale = stage->name().find("@SCALE=");
        auto fusedScaleShift = stage->name().find("FusedScaleShift_");
        auto fusedPowerShift = stage->name().find("FusedPower_");
        if (fusedPowerShift == std::string::npos) {
            fusedPowerShift = stage->name().find("fused_power");
        }
        auto addScale = stage->name().find("Add_");

        if (inputScale != std::string::npos ||
            fusedPowerShift != std::string::npos) {
            if (stage->type() == StageType::Power) {
                auto powerScale = stage->attrs().get<float>("scale");
                if (powerScale < 0.125f) {
                    removeScale = false;
                    break;
                }
            }
        }

        if (fusedScaleShift != std::string::npos ||
            addScale != std::string::npos) {
            if (stage->type() == StageType::ScaleShift) {
                auto scales = stage->input(1);

                auto content = scales->content();
                IE_ASSERT(content != nullptr);

                auto scalesVals = content->get<fp16_t>();
                IE_ASSERT(scalesVals != nullptr);

                for (int i = 0; i < scales->desc().totalDimSize(); ++i) {
                    if (ie::PrecisionUtils::f16tof32(scalesVals[i]) < 0.125f) {
                        removeScale = false;
                        break;
                    }
                }

                if (!removeScale) {
                    break;
                }
            }
        }
    }

    return removeScale;
}

class PerLayerScalePassImpl final : public Pass {
public:
    void run(const Model::Ptr& model) override;
};


int correctShift(int shift, bool firstStage, const std::string& type) {
    auto caselessEq = InferenceEngine::details::CaselessEq<std::string>();

    if (firstStage && shift > 10) {
        shift -= 8;
    }

    if (caselessEq(type, "Convolution") || caselessEq(type, "Deconvolution")) {
        shift = std::min(shift, 8);
    } else if (caselessEq(type, "FullyConnected")) {
        shift = std::min(shift, 9);
    }

    return shift;
}

int maxOutputExponent(const std::string& name, const InferenceEngine::NetworkStatsMap& stats) {
    auto node_stats_it = stats.find(name);
    IE_ASSERT(node_stats_it != stats.end());

    auto& max = node_stats_it->second->_maxOutputs;
    auto& min = node_stats_it->second->_maxOutputs;

    IE_ASSERT(max.size() > 0 && min.size() > 0);
    auto max_value = *std::max_element(max.begin(), max.end());
    auto min_value = *std::min_element(min.begin(), min.end());

    max_value = std::max(fabsf(max_value), fabsf(min_value));
    IE_ASSERT(max_value > 0);
    int exp = 0;

    // frexp fractions float into two parts:
    // [0.5, 1)* 2^exp
    // while float stores value in format
    // [1, 2) * 2^f_exp
    // which means exp returned by frexp is f_exp + 1
    frexp(max_value, &exp);
    return exp - 1;
}

void PerLayerScalePassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(analyzeWeightableLayers);

    static const int scaleToExp     = 8;  // get from config?
    static const int scaleThreshold = 1;

    auto& stats  = model->nodesStats();

    bool isGrowingOutput = checkGrowingOutput(model);

    bool firstStage = true;
    int  normalVal  = 0;

    for (const auto& stage : model->getStages()) {
        if (!isScalable(stage)) {
            continue;
        }

        auto weights = stage->input(1);

        auto content = weights->content();
        IE_ASSERT(content != nullptr);

        auto weightsVals = content->get<fp16_t>();
        IE_ASSERT(weightsVals != nullptr);

        auto exponents = calculateExponents(weightsVals, weights->desc().totalDimSize());

        int maxExp = *std::max_element(exponents.begin(), exponents.end());
        int shift  = largestExp - maxExp;

        auto meanExp = getMeanValue(exponents);
        shift        = std::min(-meanExp, shift);

        if (stats.empty()) {
            if (firstStage && shift < 4 && isGrowingOutput && weights->desc().dim(Dim::C) > 1) {
                normalVal = 5;
            }

            shift  = correctShift(shift, firstStage, stage->origLayer()->type);
            shift -= normalVal;
        } else {
            int outExp = maxOutputExponent(stage->origLayer()->name, stats);  // what if outExp == 15?
            shift      = std::min(scaleToExp - outExp, shift);
        }

        firstStage = false;
        float scale = 1;
        if (shift > scaleThreshold) {
            scale = 1 << shift;
        }

        stage->attrs().set<float>("scaleFactor", scale);
    }
}

}  // namespace

Pass::Ptr PassManager::estimateSingleNetworkScale() {
    return std::make_shared<SingleScalePassImpl>();
}

Pass::Ptr PassManager::analyzeWeightableLayers() {
    return std::make_shared<PerLayerScalePassImpl>();
}

}  // namespace vpu
