// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/multiclass_nms.hpp"

#include "common_test_utils/test_enums.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
std::string MulticlassNmsLayerTest::getTestCaseName(const testing::TestParamInfo<MulticlassNmsParams>& obj) {
    std::vector<InputShape> shapes;
    InputTypes input_types;
    int32_t nmsTopK, backgroundClass, keepTopK;
    element::Type outType;
    op::util::MulticlassNmsBase::SortResultType sortResultType;
    InputfloatVar inFloatVar;
    InputboolVar inboolVar;
    std::string targetDevice;

    std::tie(shapes, input_types, nmsTopK, inFloatVar, backgroundClass, keepTopK, outType, sortResultType, inboolVar, targetDevice) = obj.param;

    ElementType paramsPrec, roisnumPrec;
    std::tie(paramsPrec, roisnumPrec) = input_types;

    float iouThr, scoreThr, nmsEta;
    std::tie(iouThr, scoreThr, nmsEta) = inFloatVar;

    bool sortResCB, normalized;
    std::tie(sortResCB, normalized) = inboolVar;

    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }

    using ov::test::utils::operator<<;
    result << ")_paramsPrec=" << paramsPrec << "_roisnumPrec=" << roisnumPrec << "_";
    result << "nmsTopK=" << nmsTopK << "_";
    result << "iouThr=" << iouThr << "_scoreThr=" << scoreThr << "_backgroundClass=" << backgroundClass << "_";
    result << "keepTopK=" << keepTopK << "_outType=" << outType << "_";
    result << "sortResultType=" << sortResultType << "_sortResCrossBatch=" << sortResCB << "_nmsEta=" << nmsEta << "_normalized=" << normalized << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void MulticlassNmsLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();

    const auto& funcInputs = function->inputs();
    ASSERT_TRUE(funcInputs.size() == 2 || funcInputs.size() == 3) << "Expected 3 inputs or 2 inputs.";
    for (int i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor;

        if (i == 1) { // scores
            const size_t range = 1;
            const size_t start_from = 0;
            const size_t k = 1000;
            const int seed = 1;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i],
                                                             ov::test::utils::InputGenerateData(start_from, range, k, seed));
        } else if (i == 0) { // bboxes
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
        } else { // roisnum
            /* sum of rois is no larger than num_bboxes. */
            ASSERT_TRUE(targetInputStaticShapes[i].size() == 1) << "Expected shape size 1 for input roisnum, got: " << targetInputStaticShapes[i];

            // returns num random values whose sum is max_num.
            auto _generate_roisnum = [](int num, int max_num) {
                std::vector<int> array;
                std::vector<int> results(num);

                array.push_back(0);
                for (auto i = 0; i < num-1; i++) {
                    array.push_back(std::rand() % max_num);
                }
                array.push_back(max_num);

                std::sort(array.begin(), array.end());

                for (auto i = 0; i < num; i++) {
                    results[i] = array[i+1] - array[i];
                }

                return results;
            };
            auto roisnum = _generate_roisnum(targetInputStaticShapes[i][0], targetInputStaticShapes[0][1]/*num_bboxes*/);

            tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            if (tensor.get_element_type() == ov::element::i32) {
                auto dataPtr = tensor.data<int32_t>();
                for (size_t i = 0; i < roisnum.size(); i++) {
                    dataPtr[i] = static_cast<int32_t>(roisnum[i]);
                }
            } else {
                auto dataPtr = tensor.data<int64_t>();
                for (size_t i = 0; i < roisnum.size(); i++) {
                    dataPtr[i] = static_cast<int64_t>(roisnum[i]);
                }
            }
        }

        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

void MulticlassNmsLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    InputTypes input_types;
    size_t maxOutBoxesPerClass, backgroundClass, keepTopK;
    element::Type outType;

    op::util::MulticlassNmsBase::SortResultType sortResultType;

    InputfloatVar inFloatVar;
    InputboolVar inboolVar;
    ov::op::util::MulticlassNmsBase::Attributes attrs;

    std::tie(shapes, input_types, maxOutBoxesPerClass, inFloatVar, backgroundClass, keepTopK, outType, sortResultType, inboolVar, targetDevice)
        = this->GetParam();

    init_input_shapes(shapes);

    ElementType paramsPrec, roisnumPrec;
    std::tie(paramsPrec, roisnumPrec) = input_types;

    float iouThr, scoreThr, nmsEta;
    std::tie(iouThr, scoreThr, nmsEta) = inFloatVar;

    bool sortResCB, normalized;
    std::tie(sortResCB, normalized) = inboolVar;

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(paramsPrec, inputDynamicShapes.at(0)),
                                std::make_shared<ov::op::v0::Parameter>(paramsPrec, inputDynamicShapes.at(1))};
    if (inputDynamicShapes.size() > 2)
        params.push_back(std::make_shared<ov::op::v0::Parameter>(roisnumPrec, inputDynamicShapes.at(2)));

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

    std::shared_ptr<ov::Node> nms;
    if (!use_op_v8) {
        if (params.size() > 2) {
            nms = std::make_shared<ov::op::v9::MulticlassNms>(params[0], params[1], params[2], attrs);
        } else {
            nms = std::make_shared<ov::op::v9::MulticlassNms>(params[0], params[1], attrs);
        }
    } else {
        nms =  std::make_shared<ov::op::v8::MulticlassNms>(params[0], params[1], attrs);
    }

    function = std::make_shared<ov::Model>(nms, params, "MulticlassNMS");
}

void MulticlassNmsLayerTest8::SetUp() {
    use_op_v8 = true;
    MulticlassNmsLayerTest::SetUp();
}
} // namespace test
} // namespace ov
