// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/multiclass_nms.hpp"

namespace LayerTestsDefinitions {

using namespace ngraph;
using namespace InferenceEngine;
using namespace FuncTestUtils::PrecisionUtils;

std::string MulticlassNmsLayerTest::getTestCaseName(
    testing::TestParamInfo<MulticlassNmsParams> obj) {
  InputShapeParams inShapeParams;
  InputPrecisions inPrecisions;
  int32_t nmsTopK, backgroundClass, keepTopK;
  element::Type outType;

  op::util::NmsBase::SortResultType sortResultType;

  InputfloatVar inFloatVar;
  InputboolVar inboolVar;

  std::string targetDevice;

  std::tie(inShapeParams, inPrecisions, nmsTopK, inFloatVar, backgroundClass,
           keepTopK, outType, sortResultType, inboolVar, targetDevice) =
      obj.param;

  size_t numBatches, numBoxes, numClasses;
  std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

  Precision paramsPrec, maxBoxPrec, thrPrec;
  std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

  float iouThr, scoreThr, nmsEta;
  std::tie(iouThr, scoreThr, nmsEta) = inFloatVar;

  bool sortResCB, normalized;
  std::tie(sortResCB, normalized) = inboolVar;

  std::ostringstream result;
  result << "numBatches=" << numBatches << "_numBoxes=" << numBoxes
         << "_numClasses=" << numClasses << "_";
  result << "paramsPrec=" << paramsPrec << "_maxBoxPrec=" << maxBoxPrec
         << "_thrPrec=" << thrPrec << "_";
  result << "nmsTopK=" << nmsTopK << "_";
  result << "iouThr=" << iouThr << "_scoreThr=" << scoreThr
         << "_backgroundClass=" << backgroundClass << "_";
  result << "keepTopK=" << keepTopK << "_outType=" << outType << "_";
  result << "sortResultType=" << sortResultType
         << "_sortResCrossBatch=" << sortResCB << "_nmsEta=" << nmsEta
         << "_normalized=" << normalized << "_";
  result << "TargetDevice=" << targetDevice;
  return result.str();
}

void MulticlassNmsLayerTest::GenerateInputs() {
  size_t it = 0;
  for (const auto &input : cnnNetwork.getInputsInfo()) {
    const auto &info = input.second;
    Blob::Ptr blob;

    if (it == 1) {
      blob = make_blob_with_precision(info->getTensorDesc());
      blob->allocate();
      CommonTestUtils::fill_data_random_float<Precision::FP32>(blob, 1, 0,
                                                               1000);
    } else {
      blob = GenerateInput(*info);
    }
    inputs.push_back(blob);
    it++;
  }
}

void MulticlassNmsLayerTest::Compare(
    const std::vector<std::pair<ngraph::element::Type,
                                std::vector<std::uint8_t>>> &expectedOutputs,
    const std::vector<Blob::Ptr> &actualOutputs) {
  auto batchIndex = -1;
  std::vector<int32_t> numPerBatch(numBatches);
  for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1;
       outputIndex >= 0; outputIndex--) {
    const auto &actual = actualOutputs[outputIndex];
    const auto _dims = actual->getTensorDesc().getDims();
    if (_dims.size() == 1 && _dims[0] == numBatches) {
      batchIndex = outputIndex;
      auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
      IE_ASSERT(memory);
      const auto lockedMemory = memory->wmap();
      const auto actualBuffer = lockedMemory.as<const uint8_t *>();
      auto buffer = reinterpret_cast<const int32_t *>(actualBuffer);
      std::copy_n(buffer, numBatches, numPerBatch.begin());
    }
  }

  for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1;
       outputIndex >= 0; outputIndex--) {
    const auto &expected = expectedOutputs[outputIndex];
    const auto &actual = actualOutputs[outputIndex];

    // Compare Selected Outputs & Selected Indices
    if (outputIndex != batchIndex) {
      const auto &expectedBuffer = expected.second.data();
      auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
      IE_ASSERT(memory);
      const auto lockedMemory = memory->wmap();
      const auto actualBuffer = lockedMemory.as<const uint8_t *>();

      auto k = static_cast<float>(expected.first.size()) /
               actual->getTensorDesc().getPrecision().size();
      // W/A for int4, uint4
      if (expected.first == ngraph::element::Type_t::u4 ||
          expected.first == ngraph::element::Type_t::i4) {
        k /= 2;
      }
      if (outputIndex == 2) {
        if (expected.second.size() != k * actual->byteSize())
          throw std::runtime_error(
              "Expected and actual size 3rd output have different size");
      }

      const auto &precision = actual->getTensorDesc().getPrecision();
      auto expected_offset = 0;
      auto actual_offset = 0;
      for (size_t i = 0; i < numPerBatch.size(); i++) {
        auto validNums = numPerBatch[i];
        switch (precision) {
        case InferenceEngine::Precision::FP32: {
          switch (expected.first) {
          case ngraph::element::Type_t::f32:
            LayerTestsUtils::LayerTestsCommon::Compare(
                reinterpret_cast<const float *>(expectedBuffer) +
                    expected_offset * 6,
                reinterpret_cast<const float *>(actualBuffer) +
                    actual_offset * 6,
                validNums * 6, 1e-5f);
            break;
          case ngraph::element::Type_t::f64:
            LayerTestsUtils::LayerTestsCommon::Compare(
                reinterpret_cast<const double *>(expectedBuffer) +
                    expected_offset * 6,
                reinterpret_cast<const float *>(actualBuffer) +
                    actual_offset * 6,
                validNums * 6, 1e-5f);
            break;
          default:
            break;
          }

          const auto fBuffer = lockedMemory.as<const float *>();
          for (size_t tailing = validNums * 6;
               tailing < maxOutputBoxesPerBatch * 6; tailing++) {
            ASSERT_TRUE(
                std::abs(fBuffer[(actual_offset * 6 + tailing)] - -1.f) < 1e-5)
                << "Invalid default value: " << fBuffer[i]
                << " at index: " << i;
          }
          break;
        }
        case InferenceEngine::Precision::I32: {
          switch (expected.first) {
          case ngraph::element::Type_t::i32:
            LayerTestsUtils::LayerTestsCommon::Compare(
                reinterpret_cast<const int32_t *>(expectedBuffer) +
                    expected_offset,
                reinterpret_cast<const int32_t *>(actualBuffer) + actual_offset,
                validNums, 0);
            break;
          case ngraph::element::Type_t::i64:
            LayerTestsUtils::LayerTestsCommon::Compare(
                reinterpret_cast<const int64_t *>(expectedBuffer) +
                    expected_offset,
                reinterpret_cast<const int32_t *>(actualBuffer) + actual_offset,
                validNums, 0);
            break;
          default:
            break;
          }
          const auto iBuffer = lockedMemory.as<const int *>();
          for (size_t tailing = validNums; tailing < maxOutputBoxesPerBatch;
               tailing++) {
            ASSERT_TRUE(iBuffer[actual_offset + tailing] == -1)
                << "Invalid default value: " << iBuffer[i]
                << " at index: " << i;
          }
          break;
        }
        default:
          FAIL() << "Comparator for " << precision
                 << " precision isn't supported";
        }
        expected_offset += validNums;
        actual_offset += maxOutputBoxesPerBatch;
      }
    } else {
      const auto &expectedBuffer = expected.second.data();
      auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
      IE_ASSERT(memory);
      const auto lockedMemory = memory->wmap();
      const auto actualBuffer = lockedMemory.as<const uint8_t *>();

      auto k = static_cast<float>(expected.first.size()) /
               actual->getTensorDesc().getPrecision().size();
      // W/A for int4, uint4
      if (expected.first == ngraph::element::Type_t::u4 ||
          expected.first == ngraph::element::Type_t::i4) {
        k /= 2;
      }
      if (outputIndex == 2) {
        if (expected.second.size() != k * actual->byteSize())
          throw std::runtime_error(
              "Expected and actual size 3rd output have different size");
      }

      const auto &precision = actual->getTensorDesc().getPrecision();
      size_t size = expected.second.size() /
                    (k * actual->getTensorDesc().getPrecision().size());
      switch (precision) {
      case InferenceEngine::Precision::I32: {
        switch (expected.first) {
        case ngraph::element::Type_t::i32:
          LayerTestsUtils::LayerTestsCommon::Compare(
              reinterpret_cast<const int32_t *>(expectedBuffer),
              reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
          break;
        case ngraph::element::Type_t::i64:
          LayerTestsUtils::LayerTestsCommon::Compare(
              reinterpret_cast<const int64_t *>(expectedBuffer),
              reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
          break;
        default:
          break;
        }
        // TODO: test usage only, dynamic shape do not need check the tail value
        // const auto iBuffer = lockedMemory.as<const int *>();
        // for (int i = size; i < actual->size(); i++) {
        //    ASSERT_TRUE(iBuffer[i] == -1) << "Invalid default value: " <<
        //    iBuffer[i] << " at index: " << i;
        //}
        break;
      }
      default:
        FAIL() << "Comparator for " << precision
               << " precision isn't supported";
      }
    }
  }
}

void MulticlassNmsLayerTest::SetUp() {
  InputShapeParams inShapeParams;
  InputPrecisions inPrecisions;
  op::v8::MulticlassNms::Attributes attrs;
  size_t maxOutBoxesPerClass, backgroundClass, keepTopK;
  element::Type outType;

  op::util::NmsBase::SortResultType sortResultType;

  InputfloatVar inFloatVar;
  InputboolVar inboolVar;

  std::tie(inShapeParams, inPrecisions, maxOutBoxesPerClass, inFloatVar,
           backgroundClass, keepTopK, outType, sortResultType, inboolVar,
           targetDevice) = this->GetParam();

  // size_t numBatches, numBoxes, numClasses;
  std::tie(numBatches, numBoxes, numClasses) = inShapeParams;
  auto realClasses = numClasses;
  if (backgroundClass >= 0 && backgroundClass <= numClasses) {
    realClasses = realClasses - 1;
  }

  maxOutputBoxesPerClass = 0;
  if (maxOutBoxesPerClass >= 0)
    maxOutputBoxesPerClass =
        std::min(numBoxes, static_cast<size_t>(maxOutBoxesPerClass));
  else
    maxOutputBoxesPerClass = numBoxes;

  maxOutputBoxesPerBatch = maxOutputBoxesPerClass * realClasses;
  if (keepTopK >= 0)
    maxOutputBoxesPerBatch =
        std::min(maxOutputBoxesPerBatch, static_cast<size_t>(keepTopK));

  Precision paramsPrec, maxBoxPrec, thrPrec;
  std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

  float iouThr, scoreThr, nmsEta;
  std::tie(iouThr, scoreThr, nmsEta) = inFloatVar;

  bool sortResCB, normalized;
  std::tie(sortResCB, normalized) = inboolVar;

  const std::vector<size_t> boxesShape{numBatches, numBoxes, 4},
      scoresShape{numBatches, numClasses, numBoxes};
  auto ngPrc = convertIE2nGraphPrc(paramsPrec);
  auto params = builder::makeParams(ngPrc, {boxesShape, scoresShape});
  auto paramOuts = helpers::convert2OutputVector(
      helpers::castOps2Nodes<op::Parameter>(params));

  attrs.iou_threshold = iouThr;
  attrs.score_threshold = scoreThr;
  attrs.nms_eta = nmsEta;
  attrs.sort_result_type = sortResultType;
  attrs.sort_result_across_batch = sortResCB;
  attrs.output_type = outType;
  attrs.nms_top_k = maxOutBoxesPerClass;
  attrs.keep_top_k = keepTopK;
  attrs.background_class = backgroundClass;
  attrs.normalized = normalized;

  auto nms = std::make_shared<opset8::MulticlassNms>(paramOuts[0], paramOuts[1],
                                                     attrs);

  auto nms_0_identity = std::make_shared<opset5::Multiply>(
      nms->output(0), opset5::Constant::create(ngPrc, Shape{1}, {1}));
  auto nms_1_identity = std::make_shared<opset5::Multiply>(
      nms->output(1), opset5::Constant::create(outType, Shape{1}, {1}));
  auto nms_2_identity = std::make_shared<opset5::Multiply>(
      nms->output(2), opset5::Constant::create(outType, Shape{1}, {1}));
  function = std::make_shared<Function>(
      OutputVector{nms_0_identity, nms_1_identity, nms_2_identity}, params,
      "MulticlassNMS");
}

} // namespace LayerTestsDefinitions
