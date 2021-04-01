// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <limits>
#include <gtest/gtest.h>
#include "base_matcher.hpp"
#include <math.h>
#include <ie_icnn_network.hpp>

namespace Regression {
namespace Matchers {
//------------------------------------------------------------------------------
// class ObjectDetectionMatcher
//------------------------------------------------------------------------------

class ObjectDetectionMatcher : public BaseMatcher {

public:
    //Helpers
    struct DetectedObject;
    class ImageDescription;
    class NetworkAdapter;

    using ImageDescriptionPtrVect = std::vector<std::shared_ptr<ImageDescription>>;
    using ScoreFunction = std::function<ImageDescriptionPtrVect(InferenceEngine::CNNNetwork&)>;

    //Constructor
    ObjectDetectionMatcher(const RegressionConfig &config);

    //Operations
    virtual void match(const ScoreFunction&);
    void checkResult(const std::vector<ImageDescription>& desired);

    void to(const ImageDescription &desired, const std::shared_ptr<NetworkAdapter>& adapter);
    void to(const std::vector<ImageDescription>& desired, const std::shared_ptr<NetworkAdapter>& adapter);

    void to(const ImageDescription &desired, const NetworkAdapter& adapter);
    void to(const std::vector<ImageDescription>& desired, const NetworkAdapter& adapter);

private:
    //Operations
    void to(const std::vector<ImageDescription>& desired, const ScoreFunction&);
    //Data section
    ImageDescriptionPtrVect res_desc_;
};

using DetectedObject = ObjectDetectionMatcher::DetectedObject;
using ImageDescription = ObjectDetectionMatcher::ImageDescription;
using NetworkAdapter = ObjectDetectionMatcher::NetworkAdapter;

//------------------------------------------------------------------------------
// class DetectedObject
//------------------------------------------------------------------------------

struct ObjectDetectionMatcher::DetectedObject {
    //Data section
    int objectType;
    float xmin, xmax, ymin, ymax, prob;

    //Constructors
    DetectedObject(int objectType, float xmin, float ymin, float xmax, float ymax, float prob, int = -1);
    DetectedObject(const DetectedObject& other);

    static float ioU(const DetectedObject& detected_object_1_, const DetectedObject& detected_object_2_);

    //Operations
    void printObj();
};

//------------------------------------------------------------------------------
// class ImageDescription
//------------------------------------------------------------------------------

class ObjectDetectionMatcher::ImageDescription {
public:
    // Constructors
    ImageDescription(bool check_probs = false);
    ImageDescription(const std::list<DetectedObject> &alist, bool check_probs = false);
    ImageDescription(const ImageDescription& obj);

    //Operations
    static float ioUMultiple(const ImageDescription &detected_objects, const ImageDescription &desired_objects);
    void addDetectedObject(const DetectedObject& detected_obj);

    // Accessors
    inline bool checkProbs() const;
public:
    //Data section
    std::list<DetectedObject> alist;

private:
    //Data section
    bool check_probs_;
};

//------------------------------------------------------------------------------
// class NetworkAdapter
//------------------------------------------------------------------------------

class ObjectDetectionMatcher::NetworkAdapter {
public:
    //Operations
    virtual std::vector<shared_ptr<ImageDescription>> score(InferenceEngine::CNNNetwork network,
            std::shared_ptr<InferenceEngine::Core> ie,
            const std::string& deviceName,
            const std::map<std::string, std::string>& config,
            const std::vector<std::string>& images_files_names,
            bool with_reshape = false,
            bool useExportImport = false) const = 0;

    //Destructor
    virtual ~NetworkAdapter() = default;
};

//------------------------------------------------------------------------------
// Implementation of methods of class ImageDescription
//------------------------------------------------------------------------------

inline bool ImageDescription::checkProbs() const {
    return check_probs_;
}

} //  namespace matchers
} //  namespace regression