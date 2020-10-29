// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <memory>

#include <vpu/middleend/allocator/allocator.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    void run(const Model& model) override;
};

void PassImpl::run(const Model& model) {
    const auto& env = CompileEnv::get();

    //
    // Check Data requirements.
    //

    for (const auto& data : model->datas()) {
        auto topParent = data->getTopParentData();

        //
        // Memory type.
        //

        auto memoryType = topParent->memReqs();

        loopOverData(topParent, [memoryType](const Data& subData) {
            auto subMemType = subData->memReqs();
            IE_ASSERT(subMemType == memoryType);
            return DataLoopStatus::NextChild;
        });

        if (memoryType == MemoryType::CMX) {
            IE_ASSERT(topParent->location() == DataLocation::CMX);
        }

        //
        // Data <-> Data Edges.
        //

        if (auto dataEdge = data->parentDataEdge()) {
            auto parent = dataEdge->parent();
            auto child = dataEdge->child();

            Data producer, consumer;
            if (dataEdge->order() == SharedDataOrder::ChildWritesToParent) {
                producer = child;
                consumer = parent;
            } else if (dataEdge->order() == SharedDataOrder::ParentWritesToChild) {
                producer = parent;
                consumer = child;
            } else {
                VPU_THROW_EXCEPTION << "Invalid data order " << dataEdge->order();
            }

            //
            // Child must be Intermediate.
            //

            IE_ASSERT(child->usage() == DataUsage::Intermediate);

            //
            // Parent can't be Temp or Fake.
            //

            IE_ASSERT(parent->usage() != DataUsage::Temp && parent->usage() != DataUsage::Fake);

            //
            // Consumer must be accesible from the producer.
            //

            Stage connectionStage;

            for (const auto& consumerEdge : producer->consumerEdges()) {
                for (const auto& outEdge : consumerEdge->consumer()->outputEdges()) {
                    if (outEdge->output() == consumer) {
                        connectionStage = consumerEdge->consumer();
                        break;
                    }
                }

                if (connectionStage != nullptr) {
                    break;
                }
            }

            IE_ASSERT(dataEdge->connectionMode() == SharedConnectionMode::SUBGRAPH || connectionStage != nullptr);

            //
            // Connection stage must be special.
            //

            IE_ASSERT(dataEdge->connectionMode() == SharedConnectionMode::SUBGRAPH || connectionStage->category() == StageCategory::Special);

            //
            // Special checks for each mode.
            //

            if (dataEdge->mode() == SharedDataMode::ROI) {
                //
                // Check connection stage type and that parent has the largest buffer.
                //

                if (dataEdge->connectionMode() == SharedConnectionMode::SINGLE_STAGE) {
                    if (connectionStage->type() == StageType::Concat ||
                        connectionStage->type() == StageType::Expand) {
                        IE_ASSERT(producer == child);
                        IE_ASSERT(consumer == parent);
                    } else if (connectionStage->type() == StageType::Split ||
                               connectionStage->type() == StageType::Crop) {
                        IE_ASSERT(producer == parent);
                        IE_ASSERT(consumer == child);
                    } else {
                        VPU_THROW_EXCEPTION
                                << "Stage type " << connectionStage->type()
                                << " can't be used for ROI data connection";
                    }
                }

                //
                // Parent and child must have the same order.
                //

                IE_ASSERT(parent->desc().dimsOrder() == child->desc().dimsOrder());

                //
                // Offset must be valid.
                //

                for (const auto& p : dataEdge->attrs().getOrDefault<DimValues>("offset", DimValues())) {
                    IE_ASSERT(parent->desc().dimsOrder().hasDim(p.first));

                    IE_ASSERT(child->desc().dim(p.first) + p.second <= parent->desc().dim(p.first));
                }

                //
                // Check strides requirements
                //

                IE_ASSERT(checkStrides(child->desc(), parent->strides(), child->requiredStrides()));
            } else if (dataEdge->mode() == SharedDataMode::Reshape) {
                //
                // Check connection stage type.
                //

                IE_ASSERT(dataEdge->connectionMode() == SharedConnectionMode::SUBGRAPH || connectionStage->type() == StageType::Reshape);

                //
                // Parent and child must have the same data type.
                //

                IE_ASSERT(parent->desc().type() == child->desc().type());

                //
                // Parent and child must have the same number of elements.
                //

                IE_ASSERT(parent->desc().totalDimSize() == child->desc().totalDimSize());

                //
                // Parent and child must be compact.
                //

                // TODO: can we weaken this restriction?
                IE_ASSERT(parent->checkStrides(StridesRequirement::compact()));
                IE_ASSERT(child->checkStrides(StridesRequirement::compact()));
            } else {
                VPU_THROW_EXCEPTION << "Invalid shared data mode " << dataEdge->mode();
            }
        }
    }

    //
    // Check Stages requirements.
    //

    StageMap<int> stageExecIndMap;

    int stageExecInd = 0;
    for (const auto& stage : model->getStages()) {
        //
        // Check that dependencies was calculated
        //

        auto curStageInd = stage->index();
        IE_ASSERT(curStageInd >= 0);

        if (stage->category() != StageCategory::Special) {
            IE_ASSERT(curStageInd >= stageExecInd);
            stageExecIndMap[stage] = stageExecInd;
        }

        for (const auto& prevStage : stage->prevStages()) {
            auto prevStageInd = prevStage->index();
            IE_ASSERT(prevStageInd >= 0);
            IE_ASSERT(prevStageInd < curStageInd);

            if (stage->category() != StageCategory::Special && prevStage->category() != StageCategory::Special) {
                auto prevStageExecInd = stageExecIndMap.at(prevStage);
                IE_ASSERT(prevStageExecInd < stageExecInd);
            }
        }

        if (stage->category() != StageCategory::Special) {
            ++stageExecInd;
        }

        //
        // Check Data DimsOrder requirements
        //

        const auto& orderInfo = stage->propagateDataOrder();

        for (const auto& inEdge : stage->inputEdges()) {
            if (orderInfo.hasInput(inEdge)) {
                auto requiredOrder = orderInfo.getInput(inEdge);
                IE_ASSERT(inEdge->input()->desc().dimsOrder() == requiredOrder);
            }
        }
        for (const auto& outEdge : stage->outputEdges()) {
            if (orderInfo.hasOutput(outEdge)) {
                auto requiredOrder = orderInfo.getOutput(outEdge);
                IE_ASSERT(outEdge->output()->desc().dimsOrder() == requiredOrder);
            }
        }

        //
        // Check Data Strides requirements
        //

        const auto& stridesInfo = stage->getDataStridesRequirements();

        for (const auto& inEdge : stage->inputEdges()) {
            if (stridesInfo.hasInput(inEdge)) {
                auto requiredStrides = stridesInfo.getInput(inEdge);
                IE_ASSERT(inEdge->input()->checkStrides(requiredStrides));
            }
        }
        for (const auto& outEdge : stage->outputEdges()) {
            if (stridesInfo.hasOutput(outEdge)) {
                auto requiredStrides = stridesInfo.getOutput(outEdge);
                IE_ASSERT(outEdge->output()->checkStrides(requiredStrides));
            }
        }

        //
        // Check Data Batch support
        //

        const auto& batchInfo = stage->getBatchSupportInfo();

        for (const auto& inEdge : stage->inputEdges()) {
            if (batchInfo.hasInput(inEdge)) {
                auto requiredBatch = batchInfo.getInput(inEdge);

                if (requiredBatch == BatchSupport::Split) {
                    IE_ASSERT(inEdge->input()->desc().dim(Dim::N, 1) == 1);
                }
            }
        }
        for (const auto& outEdge : stage->outputEdges()) {
            if (batchInfo.hasOutput(outEdge)) {
                auto requiredBatch = batchInfo.getOutput(outEdge);

                if (requiredBatch == BatchSupport::Split) {
                    IE_ASSERT(outEdge->output()->desc().dim(Dim::N, 1) == 1);
                }
            }
        }

        //
        // Check SHAVEs requirements
        //

        auto stageSHAVEsRequirements = stage->getSHAVEsRequirements();

        if (stageSHAVEsRequirements == StageSHAVEsRequirements::NeedMax) {
            IE_ASSERT(stage->numSHAVEs() == env.resources.numSHAVEs);
        } else if (stageSHAVEsRequirements == StageSHAVEsRequirements::CanBeLimited) {
            IE_ASSERT(stage->numSHAVEs() > 0);
        } else if (stageSHAVEsRequirements == StageSHAVEsRequirements::TwoOrOne) {
            IE_ASSERT(stage->numSHAVEs() == 1 || stage->numSHAVEs() == 2);
        } else if (stageSHAVEsRequirements == StageSHAVEsRequirements::OnlyOne) {
            IE_ASSERT(stage->numSHAVEs() == 1);
        } else if (stageSHAVEsRequirements == StageSHAVEsRequirements::NotNeeded) {
            IE_ASSERT(stage->numSHAVEs() == 0);
        }

        if (const auto injectedStage = stage->injectedStage()) {
            IE_ASSERT(injectedStage->numSHAVEs() == stage->numSHAVEs());

            auto injectedReqs = injectedStage->getSHAVEsRequirements();

            if (injectedReqs == StageSHAVEsRequirements::NeedMax) {
                IE_ASSERT(injectedStage->numSHAVEs() == env.resources.numSHAVEs);
            } else if (injectedReqs == StageSHAVEsRequirements::CanBeLimited) {
                IE_ASSERT(injectedStage->numSHAVEs() > 0);
            } else if (injectedReqs == StageSHAVEsRequirements::TwoOrOne) {
                IE_ASSERT(injectedStage->numSHAVEs() == 1 || stage->numSHAVEs() == 2);
            } else if (injectedReqs == StageSHAVEsRequirements::OnlyOne) {
                IE_ASSERT(injectedStage->numSHAVEs() == 1);
            } else if (injectedReqs == StageSHAVEsRequirements::NotNeeded) {
                IE_ASSERT(injectedStage->numSHAVEs() == 0);
            }
        }

        //
        // Stage specific checks
        //

        stage->finalCheck();
    }
}

}  // namespace

Pass::Ptr PassManager::finalCheck() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
