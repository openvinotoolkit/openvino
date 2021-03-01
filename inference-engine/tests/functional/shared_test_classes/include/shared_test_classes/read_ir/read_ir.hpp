// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/single_layer/psroi_pooling.hpp"
#include "shared_test_classes/single_layer/roi_pooling.hpp"
#include "shared_test_classes/single_layer/roi_align.hpp"

namespace LayerTestsDefinitions {
class ReadIRTest : public testing::WithParamInterface<std::tuple<std::string, std::string>>,
                           public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<std::string, std::string>> &obj);

protected:
    void SetUp() override;
    void Infer() override;

private:
    std::string pathToModel;

    InferenceEngine::Blob::Ptr generateROIblob(const InferenceEngine::InputInfo &info, const std::shared_ptr<ngraph::Node> roiNode) const;

    friend void PSROIPoolingLayerTest::fillROITensor(
            float *buffer, int numROIs, int batchSize, int height, int width, int groupize,
            float spatialScale, int spatialBinsX, int spatialBinsY, const std::string &mode);
    friend void ROIAlignLayerTest::fillCoordTensor(std::vector<float>& coords, int height, int width,
                                                                          float spatialScale, int pooledRatio, int pooledH, int pooledW);
    friend void ROIAlignLayerTest::fillIdxTensor(std::vector<int>& idx, int batchSize);
    friend class TestEnvironment;
};
} // namespace LayerTestsDefinitions
