// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/stages/nms.hpp>
#include <vpu/frontend/frontend.hpp>

#include <ngraph/op/non_max_suppression.hpp>

#include <memory>
#include <set>

namespace vpu {

namespace {

class StaticShapeNMS final : public NonMaxSuppression {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<StaticShapeNMS>(*this);
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
                                 {{DataType::FP16},
                                  {DataType::FP16},
                                  {DataType::S32},
                                  {DataType::FP16},
                                  {DataType::FP16}},
                                 {{DataType::S32},
                                  {DataType::S32}});
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input1 = inputEdges()[0]->input();
        auto input2 = inputEdges()[1]->input();
        auto input3 = inputEdges()[2]->input();
        auto input4 = inputEdges()[3]->input();
        auto input5 = inputEdges()[4]->input();
        auto outputData = outputEdges()[0]->output();
        auto outputDims = outputEdges()[1]->output();

        input1->serializeBuffer(serializer);
        input2->serializeBuffer(serializer);
        input3->serializeBuffer(serializer);
        input4->serializeBuffer(serializer);
        input5->serializeBuffer(serializer);
        outputData->serializeBuffer(serializer);
        outputDims->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseStaticShapeNMS(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() >= 2 && inputs.size() <= 5,
        "StaticShapeNMS parsing failed, expected number of input is in range [2, 5], but {} provided",
        inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 2,
        "StaticShapeNMS parsing failed, expected number of outputs: 2, but {} provided",
        outputs.size());

    const auto sortResultDescending = layer->GetParamAsBool("sort_result_descending");
    const auto boxEncoding = layer->GetParamAsString("box_encoding");

    VPU_THROW_UNLESS(sortResultDescending == false,
        "StaticShapeNMS: parameter sortResultDescending=true is not supported on VPU");
    VPU_THROW_UNLESS(boxEncoding == "corner" || boxEncoding == "center",
        "StaticShapeNMS: boxEncoding currently supports only two values: \"corner\" and \"center\" "
        "while {} was provided", boxEncoding);

    DataVector tempInputs = inputs;
    for (auto fake = inputs.size(); fake < 5; fake++) {
        tempInputs.push_back(model->addFakeData());
    }

    auto stage = model->addNewStage<StaticShapeNMS>(layer->name, StageType::StaticShapeNMS, layer, tempInputs, outputs);
    stage->attrs().set<bool>("center_point_box", boxEncoding == "center");
}

}  // namespace vpu
