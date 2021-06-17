// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <set>
#include <utility>

namespace vpu {

namespace {

class TileStage final : public StageNode {
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<TileStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        auto inOrder = input->desc().dimsOrder();
        auto finalOrder = inOrder;

        orderInfo.setInput(inputEdge(0), finalOrder);
        orderInfo.setOutput(outputEdge(0), finalOrder);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::OnlyOne;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        auto axis = attrs().get<Dim>("axis");
        auto tiles = attrs().get<int>("tiles");

        auto axisInd = output->desc().dimsOrder().dimInd(axis);
        IE_ASSERT(axisInd >= 0);

        serializer.append(static_cast<int32_t>(axisInd));
        serializer.append(static_cast<int32_t>(tiles));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace
struct TileParams {
    size_t axis;
    size_t tiles;
};
TileParams getAxisAndTiles (const NodePtr& node) {
        auto tile = std::dynamic_pointer_cast<ngraph::opset1::Tile> (node);
        if (!tile) {
        }

        auto tiles_node = std::dynamic_pointer_cast<ngraph::opset1::Constant> (tile->input_value(1).get_node_shared_ptr());
        if (!tiles_node) {
        }

        auto tiles = tiles_node->cast_vector<int64_t>();
        auto input_shape_rank = tile->get_input_partial_shape(0).rank().get_length();
        int64_t cur_dim_id = tiles.size() - 1;

        IE_ASSERT(!(static_cast<int64_t>(tiles.size()) != input_shape_rank));

        // IE Tile operations supports only one axis to be tiled
        // bool already_set = false;
        // int64_t axis, tiles;
        // for (size_t i = 0; i < input_shape.size(); ++i) {
        //     if (shape[i] != 1) {
        //         if (already_set) return false;
        //         axis = i;
        //         tiles = shape[i];
        //         already_set = true;
        //     }
        // }
        //
        // if (!already_set) return false;
        auto last_node = tile->input_value(0);
        auto friendly_name = tile->get_friendly_name();

        int num_of_tile_dims = 0;
        for (auto t : tiles) {
            if (t != 1) {
                num_of_tile_dims++;
            }
        }
        // Will generate sequence of Tile operations if num_of_tile_dims != 1
        // because IE Tile operations supports only one axis to be tiled.
        // To keep op name unique will use special IE specific delimiter ':'
        // Original frameworks doesn't use such delimiter in names, so it will
        // guarantee that newly generated name like "original_name:_1" doesn't
        // match with already existed names.
        if (num_of_tile_dims > 1) {
            friendly_name += ":";
        }

        ngraph::NodeVector new_ops;
        TileParams outTileParams;
        auto tiles_it = tiles.rbegin();
        while (tiles_it != tiles.rend()) {
            int64_t tile_dim = *tiles_it;
            if (tile_dim != 1) {
                outTileParams.axis = cur_dim_id;
                outTileParams.tiles = tile_dim;
            }
            --cur_dim_id;
            ++tiles_it;
        }
        return outTileParams;
}
void FrontEnd::parseTile(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    const auto& tile = ngraph::as_type_ptr<ngraph::opset4::Tile>(node);
    VPU_THROW_UNLESS(tile != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    auto params = getAxisAndTiles(tile);
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto output = outputs[0];

    IE_ASSERT(params.axis < input->desc().numDims());

    auto perm = DimsOrder::fromNumDims(input->desc().numDims()).toPermutation();
    auto axis = perm[input->desc().numDims() - 1 -  params.axis];

    auto stage = model->addNewStage<TileStage>(tile->get_friendly_name(), StageType::Tile, tile, {input}, {output});
    stage->attrs().set("axis", axis);
    stage->attrs().set("tiles", params.tiles);
}
}  // namespace vpu
