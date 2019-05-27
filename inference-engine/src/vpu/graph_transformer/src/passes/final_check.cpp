// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <memory>

#include <vpu/allocator.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    void run(const Model::Ptr& model) override;
};

void PassImpl::run(const Model::Ptr& model) {
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

            IE_ASSERT(connectionStage != nullptr);

            //
            // Connection stage must be special.
            //

            IE_ASSERT(connectionStage->category() == StageCategory::Special);

            //
            // Special checks for each mode.
            //

            if (dataEdge->mode() == SharedDataMode::ROI) {
                //
                // Check connection stage type and that parent has the largest buffer.
                //

                if (connectionStage->type() == StageType::Concat ||
                    connectionStage->type() == StageType::Expand) {
                    IE_ASSERT(producer == child);
                    IE_ASSERT(consumer == parent);
                } else if (connectionStage->type() == StageType::Split ||
                           connectionStage->type() == StageType::Shrink) {
                    IE_ASSERT(producer == parent);
                    IE_ASSERT(consumer == child);
                } else {
                    VPU_THROW_EXCEPTION
                            << "Stage type " << connectionStage->type()
                            << " can't be used for ROI data connection";
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

                IE_ASSERT(connectionStage->type() == StageType::Reshape);

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

        auto stageDataDimsOrderMap = stage->propagateDataOrder();

        auto inputs = stage->inputs();
        auto outputs = stage->outputs();

        for (const auto& input : inputs) {
            auto it = stageDataDimsOrderMap.find(input);
            if (it != stageDataDimsOrderMap.end()) {
                auto requiredOrder = it->second;
                IE_ASSERT(input->desc().dimsOrder() == requiredOrder);
            }
        }
        for (const auto& output : outputs) {
            auto it = stageDataDimsOrderMap.find(output);
            if (it != stageDataDimsOrderMap.end()) {
                auto requiredOrder = it->second;
                IE_ASSERT(output->desc().dimsOrder() == requiredOrder);
            }
        }

        //
        // Check Data Strides requirements
        //

        auto stageDataStridesMap = stage->getDataStridesRequirements();

        for (const auto& input : inputs) {
            auto it = stageDataStridesMap.find(input);
            if (it != stageDataStridesMap.end()) {
                auto requiredStrides = it->second;
                IE_ASSERT(input->checkStrides(requiredStrides));
            }
        }
        for (const auto& output : outputs) {
            auto it = stageDataStridesMap.find(output);
            if (it != stageDataStridesMap.end()) {
                auto requiredStrides = it->second;
                IE_ASSERT(output->checkStrides(requiredStrides));
            }
        }

        //
        // Check Data Batch support
        //

        auto stageBatchSupport = stage->getBatchSupportInfo();

        for (const auto& input : inputs) {
            auto it = stageBatchSupport.find(input);
            if (it != stageBatchSupport.end()) {
                auto requiredBatch = it->second;

                if (requiredBatch == BatchSupport::Split) {
                    IE_ASSERT(input->desc().dim(Dim::N, 1) == 1);
                }
            }
        }
        for (const auto& output : outputs) {
            auto it = stageBatchSupport.find(output);
            if (it != stageBatchSupport.end()) {
                auto requiredBatch = it->second;

                if (requiredBatch == BatchSupport::Split) {
                    IE_ASSERT(output->desc().dim(Dim::N, 1) == 1);
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

        for (const auto& injectedStageEdge : stage->injectedStageEdges()) {
            auto childStage = injectedStageEdge->child();

            IE_ASSERT(childStage->numSHAVEs() == stage->numSHAVEs());

            auto injectedReqs = childStage->getSHAVEsRequirements();

            if (injectedReqs == StageSHAVEsRequirements::NeedMax) {
                IE_ASSERT(childStage->numSHAVEs() == env.resources.numSHAVEs);
            } else if (injectedReqs == StageSHAVEsRequirements::CanBeLimited) {
                IE_ASSERT(childStage->numSHAVEs() > 0);
            } else if (injectedReqs == StageSHAVEsRequirements::TwoOrOne) {
                IE_ASSERT(childStage->numSHAVEs() == 1 || stage->numSHAVEs() == 2);
            } else if (injectedReqs == StageSHAVEsRequirements::OnlyOne) {
                IE_ASSERT(childStage->numSHAVEs() == 1);
            } else if (injectedReqs == StageSHAVEsRequirements::NotNeeded) {
                IE_ASSERT(childStage->numSHAVEs() == 0);
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
