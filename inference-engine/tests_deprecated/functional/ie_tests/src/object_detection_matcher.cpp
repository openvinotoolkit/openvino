// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "object_detection_matcher.hpp"
#include <legacy/details/ie_cnn_network_iterator.hpp>

#include <algorithm>

using namespace Regression::Matchers;

namespace Regression {
namespace Matchers {

using DetectedObject = ObjectDetectionMatcher::DetectedObject;
using ImageDescription = ObjectDetectionMatcher::ImageDescription;
using namespace InferenceEngine;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ObjectDetectionMatcher::DetectedObject //////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ObjectDetectionMatcher::DetectedObject::DetectedObject(int objectType,
                                                       float xmin,
                                                       float ymin,
                                                       float xmax,
                                                       float ymax,
                                                       float prob,
                                                       int)
        : objectType(objectType), xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), prob(prob) {
}

ObjectDetectionMatcher::DetectedObject::DetectedObject(const DetectedObject &other) {
    this->objectType = other.objectType;
    this->xmin = other.xmin;
    this->xmax = other.xmax;
    this->ymin = other.ymin;
    this->ymax = other.ymax;
    this->prob = other.prob;
}

float ObjectDetectionMatcher::DetectedObject::ioU(const DetectedObject &detected_object_1_,
                                                  const DetectedObject &detected_object_2_) {
    // Add small space to eliminate empty squares
    float epsilon = 1e-3;

    DetectedObject detectedObject1(detected_object_1_.objectType,
                                   detected_object_1_.xmin - epsilon,
                                   detected_object_1_.ymin - epsilon,
                                   detected_object_1_.xmax + epsilon,
                                   detected_object_1_.ymax + epsilon, detected_object_1_.prob);
    DetectedObject detectedObject2(detected_object_2_.objectType,
                                   detected_object_2_.xmin - epsilon,
                                   detected_object_2_.ymin - epsilon,
                                   detected_object_2_.xmax + epsilon,
                                   detected_object_2_.ymax + epsilon, detected_object_2_.prob);

    if (detectedObject1.objectType != detectedObject2.objectType) {
        // objects are different, so the result is 0
        return 0.0f;
    }

    if (detectedObject1.xmax < detectedObject1.xmin) return 0.0;
    if (detectedObject1.ymax < detectedObject1.ymin) return 0.0;
    if (detectedObject2.xmax < detectedObject2.xmin) return 0.0;
    if (detectedObject2.ymax < detectedObject2.ymin) return 0.0;

    float xmin = (std::max)(detectedObject1.xmin, detectedObject2.xmin);
    float ymin = (std::max)(detectedObject1.ymin, detectedObject2.ymin);
    float xmax = (std::min)(detectedObject1.xmax, detectedObject2.xmax);
    float ymax = (std::min)(detectedObject1.ymax, detectedObject2.ymax);
    // intersection
    float intr;

    if ((xmax >= xmin) && (ymax >= ymin)) {
        intr = (xmax - xmin) * (ymax - ymin);
    } else {
        intr = 0.0f;
    }

    // union
    float square1 = (detectedObject1.xmax - detectedObject1.xmin) * (detectedObject1.ymax - detectedObject1.ymin);
    float square2 = (detectedObject2.xmax - detectedObject2.xmin) * (detectedObject2.ymax - detectedObject2.ymin);

    float unn = square1 + square2 - intr;

    return float(intr) / unn;
}

void ObjectDetectionMatcher::DetectedObject::printObj() {
    printf("[%p] objectType=%d, xmin=%f, xmax=%f, ymin=%f, ymax=%f, prob=%f\n",
           this,
           objectType,
           xmin,
           xmax,
           ymin,
           ymax,
           prob);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ObjectDetectionMatcher::ImageDescription ////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ObjectDetectionMatcher::ImageDescription::ImageDescription(bool check_probs) :
        check_probs_(check_probs) {
}

ObjectDetectionMatcher::ImageDescription::ImageDescription(const std::list<DetectedObject> &alist, bool check_probs)
        : alist(alist), check_probs_(check_probs) {
}

ObjectDetectionMatcher::ImageDescription::ImageDescription(const ImageDescription &obj) :
        check_probs_(obj.checkProbs()) {
    this->alist = obj.alist;
}

float ObjectDetectionMatcher::ImageDescription::ioUMultiple(const ImageDescription &detected_objects,
                                                            const ImageDescription &desired_objects) {

    const ImageDescription *detectedObjectsSmall, *detectedObjectsBig;
    bool check_probs = desired_objects.checkProbs();

    if (detected_objects.alist.size() < desired_objects.alist.size()) {
        detectedObjectsSmall = &detected_objects;
        detectedObjectsBig = &desired_objects;
    } else {
        detectedObjectsSmall = &desired_objects;
        detectedObjectsBig = &detected_objects;
    }

    std::list<DetectedObject> doS = detectedObjectsSmall->alist;
    std::list<DetectedObject> doB = detectedObjectsBig->alist;

    float fullScore = 0.0f;
    while (doS.size() > 0) {
        float score = 0.0f;
        std::list<DetectedObject>::iterator bestJ = doB.end();
        for (auto j = doB.begin(); j != doB.end(); j++) {
            float curscore = DetectedObject::ioU(*doS.begin(), *j);
            if (score < curscore) {
                score = curscore;
                bestJ = j;
            }
        }

        float coeff = 1.0;
        if (check_probs) {
            if (bestJ != doB.end()) {
                DetectedObject test = *bestJ;
                DetectedObject test1 = *doS.begin();
                float min = std::min((*bestJ).prob, (*doS.begin()).prob);
                float max = std::max((*bestJ).prob, (*doS.begin()).prob);

                coeff = min / max;
            }
        }

        doS.pop_front();
        if (bestJ != doB.end()) doB.erase(bestJ);
        fullScore += coeff * score;
    }
    fullScore /= detectedObjectsBig->alist.size();

    return fullScore;
}

void ObjectDetectionMatcher::ImageDescription::addDetectedObject(const DetectedObject &detected_obj) {
    alist.push_back(detected_obj);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ObjectDetectionMatcher::ObjectDetectionMatcher //////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ObjectDetectionMatcher::ObjectDetectionMatcher(const RegressionConfig &config)
        : BaseMatcher(config) {
}

void ObjectDetectionMatcher::match(const ScoreFunction& score_function) {
    // Read network
    string binFileName = testing::FileUtils::fileNameNoExt(config._path_to_models) + ".bin";
    auto cnnNetwork = config.ie_core->ReadNetwork(config._path_to_models, binFileName);

    if (config._reshape) {
        auto inputShapes = cnnNetwork.getInputShapes();
        for (auto & shape : inputShapes) {
            shape.second[0] = config.batchSize;
        }

        cnnNetwork.reshape(inputShapes);
    } else if (config.batchSize != 1) {
        cnnNetwork.setBatchSize(config.batchSize);
    }

    res_desc_ = score_function(cnnNetwork);

    if (res_desc_.size() != config.batchSize) {
        FAIL() << "[ERROR]: Result batch size is not equal to initial.";
    }
}

void ObjectDetectionMatcher::checkResult(const std::vector<ImageDescription> &desired) {
    if ((desired.size() < config.batchSize) || (res_desc_.size() != config.batchSize)) {
        FAIL() << "[ERROR]: Number of ImageDescription objects less then batch size or result batch size is not equal to initial.\n"
               << "Batch size: " << config.batchSize << "; Expected outputs number: " << desired.size()
               << "; Result number: " << res_desc_.size();
    }
    string sError;
    for (int i = 0; i < config.batchSize; i++) {
        double iou = ImageDescription::ioUMultiple(*res_desc_[i], desired[i]);
        double minimalScore = 1.0 - config.nearValue;
        if (iou < minimalScore) {
            sError += "[ERROR]: Batch #" + std::to_string(i) + ". Similarity is too low: " + std::to_string(iou)
                      + ". Expected " + std::to_string(minimalScore) + "\n";
        } else {
            std::cout << "Batch #" << i << ". Similarity " << iou << " is above the expected value: " << minimalScore
                      << std::endl;
        }
    }

    if (!sError.empty()) {
        FAIL() << sError;
    }
}

void ObjectDetectionMatcher::to(const ImageDescription &desired, const std::shared_ptr<NetworkAdapter>& adapter) {
    std::vector<ImageDescription> desired_vector = {desired};
    ASSERT_NO_FATAL_FAILURE(to(desired_vector, adapter));
}

void ObjectDetectionMatcher::to(const std::vector<ImageDescription> &desired,
                                const std::shared_ptr<NetworkAdapter>& adapter) {
    to(desired, [&](CNNNetwork & network) -> ImageDescriptionPtrVect {
        return adapter->score(network,
                              config.ie_core,
                              config._device_name,
                              config.plugin_config,
                              config._paths_to_images,
                              config._reshape,
                              config.useExportImport);
    });
}

void ObjectDetectionMatcher::to(const ImageDescription &desired, const NetworkAdapter& adapter) {
    std::vector<ImageDescription> desired_vector = {desired};
    ASSERT_NO_FATAL_FAILURE(to(desired_vector, adapter));
}

void ObjectDetectionMatcher::to(const std::vector<ImageDescription> &desired,
                                const NetworkAdapter& adapter) {
    to(desired, [&](CNNNetwork& network) -> ImageDescriptionPtrVect {
        return adapter.score(network,
                             config.ie_core,
                             config._device_name,
                             config.plugin_config,
                             config._paths_to_images,
                             config._reshape);
    });
}

void ObjectDetectionMatcher::to(const std::vector<ImageDescription> &desired, const ScoreFunction& score_function) {
    // ASSERT_NO_FATAL_FAILURE(checkImgNumber());
    ASSERT_NO_FATAL_FAILURE(match(score_function));
    if (desired.size() < config.batchSize) {
        std::cout << "Number of ImageDescription objects less then batch size" << std::endl;
        std::vector<ImageDescription> newRef;
        for (int i = 0; i < config.batchSize; i++) {
            newRef.push_back(desired[0]);
        }
        ASSERT_NO_FATAL_FAILURE(checkResult(newRef));
    } else {
        ASSERT_NO_FATAL_FAILURE(checkResult(desired));
    }
}

} //  namespace matchers
} //  namespace regression