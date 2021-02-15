// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

namespace {

class AddCopyForOutputsInsideNetwork : public vpu::GraphTransformerTest {
protected:
    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());
        ASSERT_NO_FATAL_FAILURE(InitCompileEnv());
        ASSERT_NO_FATAL_FAILURE(InitPipeline());

        m_testModel = CreateTestModel();
    }

    void CheckOutputs(const vpu::DataMap<vpu::Data>& parentShapes = {}, const vpu::DataMap<vpu::DataVector>& childShapes = {}) {
        for (const auto& output : m_testModel.getOutputs()) {
            ASSERT_TRUE(output->consumerEdges().empty());
            ASSERT_TRUE((parentShapes.count(output) == 0 && output->parentDataToShapeEdge() == nullptr) ||
                (parentShapes.count(output) == 1 && output->parentDataToShapeEdge()->parent() == parentShapes.at(output)));
            ASSERT_FALSE((childShapes.count(output) == 0) ^ output->childDataToShapeEdges().empty());
            if (output->childDataToShapeEdges().empty()) {
                continue;
            }

            vpu::DataVector actualChildDataObjects;
            const auto& actualChildShapesEdges = output->childDataToShapeEdges();
            std::transform(actualChildShapesEdges.begin(), actualChildShapesEdges.end(), std::back_inserter(actualChildDataObjects),
                [](const vpu::DataToShapeAllocation& edge) { return edge->child(); });
            ASSERT_EQ(actualChildDataObjects, childShapes.at(output));
        }
    }

    void Compile() {
        m_pipeline.run(m_testModel.getBaseModel());
    }

    void InitPipeline() {
        m_pipeline = vpu::PassSet();
        m_pipeline.addPass(passManager->dumpModel("before-addCopyForOutputsInsideNetwork"));
        m_pipeline.addPass(passManager->addCopyForOutputsInsideNetwork());
        m_pipeline.addPass(passManager->dumpModel("after-addCopyForOutputsInsideNetwork"));
    }

protected:
    vpu::PassSet m_pipeline;
    vpu::TestModel m_testModel;

    const vpu::DataDesc m_defaultDescriptor = {1};
};

TEST_F(AddCopyForOutputsInsideNetwork, DynamicOutputWithoutConsumer) {
    m_testModel.createInputs({m_defaultDescriptor, m_defaultDescriptor, m_defaultDescriptor, m_defaultDescriptor});
    m_testModel.createOutputs({m_defaultDescriptor, m_defaultDescriptor});
    m_testModel.addStage({vpu::InputInfo::fromNetwork(0), vpu::InputInfo::fromNetwork(1)}, {vpu::OutputInfo::fromNetwork(0)});

    auto shapeProducer = m_testModel.addStage(
        {vpu::InputInfo::fromPrevStage(0, 0), vpu::InputInfo::constant(m_defaultDescriptor)},
        {vpu::OutputInfo::intermediate(m_defaultDescriptor)});

    m_testModel.addStage(
        {
            vpu::InputInfo::fromPrevStage(1, 0),
            vpu::InputInfo::fromNetwork(2),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::fromNetwork(3),
        },
        {vpu::OutputInfo::fromNetwork(1)});

    ASSERT_NO_THROW(m_testModel.getBaseModel()->connectDataWithShape(shapeProducer->output(0), m_testModel.getOutputs().back()));
    ASSERT_NO_THROW(Compile());
    CheckOutputs({{m_testModel.getOutputs().back(), shapeProducer->output(0)}});
}

TEST_F(AddCopyForOutputsInsideNetwork, StaticOutputWithConsumer) {
    m_testModel.createInputs({m_defaultDescriptor, m_defaultDescriptor});
    m_testModel.createOutputs({m_defaultDescriptor, m_defaultDescriptor});

    m_testModel.addStage(
        {
            vpu::InputInfo::fromNetwork(0),
            vpu::InputInfo::fromNetwork(1),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::constant(m_defaultDescriptor)
        },
        {
            vpu::OutputInfo::fromNetwork(0),
            vpu::OutputInfo::fromNetwork(1)});

    const auto shapeProducer = m_testModel.addStage(
        {
            vpu::InputInfo::fromPrevStage(0, 0),
            vpu::InputInfo::constant(m_defaultDescriptor)
        },
        {
            vpu::OutputInfo::intermediate(m_defaultDescriptor)
        });

    m_testModel.getBaseModel()->connectDataWithShape(shapeProducer->output(0), m_testModel.getOutputs().back());

    ASSERT_NO_THROW(Compile());
    CheckOutputs({{m_testModel.getOutputs().back(), shapeProducer->output(0)}});
}

TEST_F(AddCopyForOutputsInsideNetwork, DynamicOutputWithConsumer) {
    m_testModel.createInputs({m_defaultDescriptor, m_defaultDescriptor, m_defaultDescriptor});
    m_testModel.createOutputs({m_defaultDescriptor, m_defaultDescriptor, m_defaultDescriptor});

    m_testModel.addStage(
        {
            vpu::InputInfo::fromNetwork(0),
            vpu::InputInfo::fromNetwork(1),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::constant(m_defaultDescriptor)
        },
        {
            vpu::OutputInfo::fromNetwork(0),
            vpu::OutputInfo::fromNetwork(1)});

    const auto shapeProducer = m_testModel.addStage(
        {
            vpu::InputInfo::fromPrevStage(0, 0),
            vpu::InputInfo::constant(m_defaultDescriptor)
        },
        {
            vpu::OutputInfo::intermediate(m_defaultDescriptor)
        });

    m_testModel.getBaseModel()->connectDataWithShape(shapeProducer->output(0), m_testModel.getOutputs()[1]);

    m_testModel.addStage(
        {
            vpu::InputInfo::fromNetwork(2),
            vpu::InputInfo::fromPrevStage(1, 0),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::fromPrevStage(0, 1),
        },
        {vpu::OutputInfo::fromNetwork(2)});

    ASSERT_NO_THROW(Compile());
    CheckOutputs({{m_testModel.getOutputs()[1], shapeProducer->output(0)}});
}

TEST_F(AddCopyForOutputsInsideNetwork, StaticOutputWithConsumerAndDynamicOutputWithConsumer) {
    m_testModel.createInputs({m_defaultDescriptor, m_defaultDescriptor, m_defaultDescriptor, m_defaultDescriptor});
    m_testModel.createOutputs({m_defaultDescriptor, m_defaultDescriptor, m_defaultDescriptor, m_defaultDescriptor});

    m_testModel.addStage(
        {
            vpu::InputInfo::fromNetwork(0),
            vpu::InputInfo::fromNetwork(1),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::constant(m_defaultDescriptor)
        },
        {
            vpu::OutputInfo::fromNetwork(0),
            vpu::OutputInfo::fromNetwork(1)});

    const auto shapeProducer_0 = m_testModel.addStage(
        {
            vpu::InputInfo::fromPrevStage(0, 0),
            vpu::InputInfo::constant(m_defaultDescriptor)
        },
        {
            vpu::OutputInfo::intermediate(m_defaultDescriptor)
        });

    m_testModel.getBaseModel()->connectDataWithShape(shapeProducer_0->output(0), m_testModel.getOutputs()[1]);

    m_testModel.addStage(
        {
            vpu::InputInfo::fromNetwork(2),
            vpu::InputInfo::fromNetwork(3),
        },
        {
            vpu::OutputInfo::fromNetwork(3)
        });

    const auto shapeProducer_1 = m_testModel.addStage(
        {
            vpu::InputInfo::fromPrevStage(2, 0),
            vpu::InputInfo::constant(m_defaultDescriptor)
        },
        {
            vpu::OutputInfo::intermediate(m_defaultDescriptor)
        });

    m_testModel.addStage(
        {
            vpu::InputInfo::fromPrevStage(3, 0),
            vpu::InputInfo::fromPrevStage(1, 0),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::constant(m_defaultDescriptor),
            vpu::InputInfo::fromPrevStage(0, 1),
        },
        {vpu::OutputInfo::fromNetwork(2)});

    m_testModel.getBaseModel()->connectDataWithShape(shapeProducer_1->output(0), m_testModel.getOutputs()[2]);

    ASSERT_NO_THROW(Compile());
    CheckOutputs({
        {m_testModel.getOutputs()[1], shapeProducer_0->output(0)},
        {m_testModel.getOutputs()[2], shapeProducer_1->output(0)}});
}

} // namespace
