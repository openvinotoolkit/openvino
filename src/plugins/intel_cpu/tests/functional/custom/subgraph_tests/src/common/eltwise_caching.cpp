// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Motivation:
// In a dynamic scenario, depending on the input shapes for the current node, we can either generate a new jit kernel or get an existing one from the cache.
// But the current single layer tests do not allow checking the case when the same kernel can be used for different nodes.
// This subgraph test contains 2 eltwise nodes and allows us to check this case.
// This subgraph can also contain the FakeQuantize nodes, because their shapes affect the result of caching and cache lookups (post op for Eltwise).

//  -----------              -----------    -----------              -----------
//  |input 0.0|              |input 0.1|    |input 1.0|              |input 1.1|
//  -----------              -----------    -----------              -----------
//       |                        |              |                        |
//  ------------------------------------    ------------------------------------
//  |            eltwise 0             |    |            eltwise 1             |
//  ------------------------------------    ------------------------------------
//                   |                                       |
//  ------------------------------------    ------------------------------------
//  |FQ 0 (if withQuantization == true)|    |FQ 1 (if withQuantization == true)|
//  ------------------------------------    ------------------------------------
//                   |                                       |
//                   |                      ------------------------------------
//                   |                      | reshape (if needReshape == true) |
//                   |                      ------------------------------------
//                   |                                       |
//  ----------------------------------------------------------------------------
//  |                                 concat                                   |
//  ----------------------------------------------------------------------------
//                                       |
//                                   --------
//                                   |output|
//                                   --------

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/filter_cpu_info.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "internal_properties.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

using namespace CPUTestUtils;

namespace ov {
namespace test {
using namespace ov::test::utils;

using InputShapesTuple = std::tuple<
        std::vector<InputShape>,            // eltwise input shapes
        std::vector<std::vector<size_t>>,   // fq input shapes
        std::vector<int32_t>                // reshape shape
>;

typedef std::tuple<
        InputShapesTuple,                   // eltwise and fq input shapes
        std::vector<ElementType>,           // Input precisions
        std::vector<EltwiseTypes>,          // Eltwise operations
        bool,                               // With quantization
        bool,                               // Need reshape
        bool,                               // Enforce Snippets
        std::string,                        // Device name
        CPUSpecificParams
> EltwiseCacheTestParams;

class EltwiseCacheTest : public testing::WithParamInterface<EltwiseCacheTestParams>,
                         virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<EltwiseCacheTestParams> &obj) {
        InputShapesTuple inputShapesTuple;
        std::vector<ElementType> inputPrecisions;
        std::vector<EltwiseTypes> eltwiseOpTypes;
        bool withQuantization;
        bool needReshape;
        bool enforceSnippets;
        std::string targetName;
        CPUSpecificParams cpuParams;
        std::tie(inputShapesTuple, inputPrecisions, eltwiseOpTypes, withQuantization, needReshape, enforceSnippets,
                 targetName, cpuParams) = obj.param;

        std::vector<InputShape> eltwiseInputShapes;
        std::vector<std::vector<size_t>> fqInputShapes;
        std::vector<int32_t> reshapeShape;
        std::tie(eltwiseInputShapes, fqInputShapes, reshapeShape) = inputShapesTuple;

        std::ostringstream results;

        results << "IS=(";
        for (const auto& shape : eltwiseInputShapes) {
            results << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        results << ")_TS=(";
        for (const auto& shape : eltwiseInputShapes) {
            for (const auto& item : shape.second) {
                results << ov::test::utils::vec2str(item) << "_";
            }
        }
        if (withQuantization) {
            results << ")_FQS=(";
            for (const auto& shape : fqInputShapes) {
                results << ov::test::utils::vec2str(shape) << "_";
            }
        }
        if (needReshape) {
            results << ")_RS=(";
            results << ov::test::utils::vec2str(reshapeShape) << "_";
        }
        results << ")_";
        for (size_t i = 0; i < inputPrecisions.size(); i++) {
            results << "InPRC" << std::to_string(i) << "=" << inputPrecisions[i] << "_";
        }
        for (size_t i = 0; i < eltwiseOpTypes.size(); i++) {
            results << "Op" << std::to_string(i) << "=" << eltwiseOpTypes[i] << "_";
        }
        results << "WithQuant=" << withQuantization << "_";
        results << "enforceSnippets=" << enforceSnippets << "_";
        results << "targetDevice=" << targetName;

        results << CPUTestsBase::getTestCaseName(cpuParams);

        return results.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 1;
            in_data.range = 10;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        abs_threshold = 0.1f;

        InputShapesTuple inputShapesTuple;
        std::vector<ElementType> inputPrecisions;
        std::vector<EltwiseTypes> eltwiseOpTypes;
        bool withQuantization;
        bool needReshape;
        bool enforceSnippets;
        CPUSpecificParams cpuParams;
        std::tie(inputShapesTuple, inputPrecisions, eltwiseOpTypes, withQuantization, needReshape, enforceSnippets,
                 targetDevice, cpuParams) = this->GetParam();

        std::vector<InputShape> eltwiseInputShapes;
        std::vector<std::vector<size_t>> fqInputShapes;
        std::vector<int32_t> reshapeShape;
        std::tie(eltwiseInputShapes, fqInputShapes, reshapeShape) = inputShapesTuple;

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        init_input_shapes(eltwiseInputShapes);

        if (!enforceSnippets) {
            configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
        }

        ov::ParameterVector paramVec;
        std::vector<std::shared_ptr<ov::Node>> inputNodes;

        for (size_t i = 0; i < inputDynamicShapes.size(); i++) {
            paramVec.push_back(std::make_shared<ov::op::v0::Parameter>(inputPrecisions[i], inputDynamicShapes[i]));
            inputNodes.push_back(paramVec.back());
        }

        auto lastNode0 = utils::make_eltwise(paramVec[0], paramVec[1], eltwiseOpTypes[0]);
        lastNode0->get_rt_info() = getCPUInfo();
        auto lastNode1 = utils::make_eltwise(paramVec[2], paramVec[3], eltwiseOpTypes[1]);
        lastNode1->get_rt_info() = getCPUInfo();
        if (withQuantization) {
            lastNode0 = ov::test::utils::make_fake_quantize(lastNode0, ov::element::Type(ov::element::Type_t::f32),
                                                          256, fqInputShapes[0]);
            lastNode1 = ov::test::utils::make_fake_quantize(lastNode1, ov::element::Type(ov::element::Type_t::f32),
                                                          256, fqInputShapes[1]);
        }
        if (needReshape) {
            auto reshapeConstNode = ov::op::v0::Constant::create(ov::element::Type(ov::element::Type_t::i32),
                                                                 ov::Shape{reshapeShape.size()}, reshapeShape);
            lastNode1 = std::make_shared<ov::op::v1::Reshape>(lastNode1, reshapeConstNode, false);
        }
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{lastNode0, lastNode1}, 0);
        function = std::make_shared<ov::Model>(concat, paramVec, "eltwise_cache");
    }
};

TEST_P(EltwiseCacheTest, CompareWithRefs) {
    run();
}

namespace {

std::vector<bool> enforceSnippets = {
    false, true
};

std::vector<std::vector<ElementType>> inputPrecisions {
        { ElementType::f32, ElementType::f32, ElementType::f32, ElementType::f32 }
};

std::vector<std::vector<EltwiseTypes>> eltwiseOps = {
        { EltwiseTypes::ADD, EltwiseTypes::ADD }
};

CPUSpecificParams cpuParams_empty = {{}, {}, {}, {}};

std::vector<InputShapesTuple> inputShapes_2D_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1},
                // target
                {
                    {5, 6}, // miss
                    {8, 6}, // no need serach in cache
                    {5, 6}, // miss
                    {5, 1}, // hit
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1},
                // target
                {
                    {5, 6},
                    {8, 6},
                    {5, 1},
                    {5, 6},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, -1},
                // target
                {
                    {7, 6}, // hit
                    {9, 6}, // no need search in cache
                    {7, 1}, // miss
                    {7, 6}, // hit
                }
            },
            // in1.1
            {
                // dynamic
                {-1, -1},
                // target
                {
                    {7, 6},
                    {9, 6},
                    {7, 6},
                    {7, 1},
                }
            }
        },
        // fq shapes
        {
            {1, 6},
            {1, 6},
        },
        // reshape shape
        {}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_2D_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_2D_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false, true),
                                ::testing::Values(false),
                                ::testing::ValuesIn(enforceSnippets),
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::Values(cpuParams_empty)),
                        EltwiseCacheTest::getTestCaseName);

// different last dim
std::vector<InputShapesTuple> inputShapes_2D_diff_last_dim_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1},
                // target
                {
                    {5, 6} // miss
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1},
                // target
                {
                    {5, 6}
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, -1},
                // target
                {
                    {14, 3} // hit
                }
            },
            // in1.1
            {
                // dynamic
                {-1, -1},
                // target
                {
                    {14, 3}
                }
            }
        },
        // fq shapes
        {
            {1, 6},
            {1, 3},
        },
        // reshape shape
        {-1, 6}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_2D_diff_last_dim_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_2D_diff_last_dim_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false, true),
                                ::testing::Values(true),
                                ::testing::ValuesIn(enforceSnippets),
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::Values(cpuParams_empty)),
                        EltwiseCacheTest::getTestCaseName);

// 1D + 2D combo
std::vector<InputShapesTuple> inputShapes_1D_2D_combo_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1},
                // target
                {
                    {1, 6}, // miss
                    {1, 6}, // miss
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1},
                // target
                {
                    {1, 6},
                    {1, 1},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1},
                // target
                {
                    {6}, // hit
                    {6}, // hit
                }
            },
            // in1.1
            {
                // dynamic
                {-1},
                // target
                {
                    {6},
                    {1},
                }
            }
        },
        // fq shapes
        {},
        // reshape shape
        {-1, 6}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_1D_2D_combo_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_1D_2D_combo_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(true),
                                ::testing::ValuesIn(enforceSnippets),
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::Values(cpuParams_empty)),
                        EltwiseCacheTest::getTestCaseName);


// 3D
CPUSpecificParams cpuParams_3D_planar = {{ncw, ncw}, {ncw}, {}, {}};
CPUSpecificParams cpuParams_3D_blocked = {{nCw16c, nCw16c}, {nCw16c}, {}, {}};
CPUSpecificParams cpuParams_3D_nspc = {{nwc, nwc}, {nwc}, {}, {}};
std::vector<CPUSpecificParams> cpuParams_3D_blocked_vec = {cpuParams_3D_blocked};

std::vector<InputShapesTuple> inputShapes_3D_planar_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1, -1},
                // target
                {
                    {2, 3, 4}, // miss
                    {2, 3, 4}, // miss
                    {2, 3, 1}, // hit
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1, -1},
                // target
                {
                    {2, 3, 4},
                    {2, 3, 1},
                    {2, 3, 4},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, -1, -1},
                // target
                {
                    {2, 6, 8}, // hit
                    {2, 6, 1}, // miss
                    {2, 6, 1}, // no need search in cache
                }
            },
            // in1.1
            {
                // dynamic
                {-1, -1, -1},
                // target
                {
                    {2, 6, 8},
                    {2, 6, 8},
                    {2, 6, 16},
                }
            }
        },
        // fq shapes
        {
            {1, 3, 1}, {1, 6, 1}
        },
        // reshape shape
        {-1, 3, 4}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_3D_planar_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_3D_planar_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false, true), // withQuantization
                                ::testing::Values(true), // needReshape
                                ::testing::ValuesIn(enforceSnippets),
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::Values(cpuParams_3D_planar)),
                        EltwiseCacheTest::getTestCaseName);

std::vector<InputShapesTuple> inputShapes_3D_blocked_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, {2, 64}, -1},
                // target
                {
                    {2, 24, 5}, // miss
                    {2, 24, 5}, // no need search in cache
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, {2, 64}, -1},
                // target
                {
                    {2, 24, 5},
                    {2, 24, 1},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, {2, 64}, -1},
                // target
                {
                    {2, 48, 10}, // hit
                    {2, 48, 1},  // no need search in cache
                }
            },
            // in1.1
            {
                // dynamic
                {-1, {2, 64}, -1},
                // target
                {
                    {2, 48, 10},
                    {2, 48, 10},
                }
            }
        },
        // fq shapes
        {},
        // reshape shape
        {-1, 24, 5}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_3D_blocked_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_3D_blocked_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(true),
                                ::testing::Values(false),  // CPU Plugin doesn't support non-planar layout for Subgraphs
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_3D_blocked_vec))),
                        EltwiseCacheTest::getTestCaseName);

std::vector<InputShapesTuple> inputShapes_3D_nspc_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1, -1},
                // target
                {
                    {2, 3, 4}, // miss
                    {2, 3, 4}, // miss
                    {2, 1, 4}, // hit
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1, -1},
                // target
                {
                    {2, 3, 4},
                    {2, 1, 4},
                    {2, 3, 4},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, -1, -1},
                // target
                {
                    {2, 6, 8}, // hit
                    {2, 1, 8}, // miss
                    {2, 1, 8}, // no need search in cache
                }
            },
            // in1.1
            {
                // dynamic
                {-1, -1, -1},
                // target
                {
                    {2, 6, 8},
                    {2, 6, 8},
                    {2, 12, 8},
                }
            }
        },
        // fq shapes
        {},
        // reshape shape
        {-1, 3, 4}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_3D_nspc_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_3D_nspc_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(true),
                                ::testing::Values(false),  // CPU Plugin doesn't support non-planar layout for Subgraphs
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(filterCPUSpecificParams({cpuParams_3D_nspc}))),
                        EltwiseCacheTest::getTestCaseName);


// 4D
CPUSpecificParams cpuParams_4D_planar = {{nchw, nchw}, {nchw}, {}, {}};
CPUSpecificParams cpuParams_4D_blocked = {{nChw16c, nChw16c}, {nChw16c}, {}, {}};
CPUSpecificParams cpuParams_4D_nspc = {{nhwc, nhwc}, {nhwc}, {}, {}};
std::vector<CPUSpecificParams> cpuParams_4D_blocked_vec = {cpuParams_4D_blocked};

std::vector<InputShapesTuple> inputShapes_4D_planar_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 3, 4, 5},
                    {2, 3, 4, 5},
                    {2, 3, 4, 1},
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 3, 4, 5},
                    {2, 3, 4, 1},
                    {2, 3, 4, 5},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 6, 4, 10},
                    {2, 6, 4, 1},
                    {2, 6, 4, 1},
                }
            },
            // in1.1
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 6, 4, 10},
                    {2, 6, 4, 10},
                    {2, 6, 4, 20},
                }
            }
        },
        // fq shapes
        {
            {1, 3, 1, 1}, {1, 6, 1, 1}
        },
        // reshape shape
        {-1, 3, 4, 5}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_4D_planar_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_4D_planar_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false, true), // withQuantization
                                ::testing::Values(true), // needReshape
                                ::testing::ValuesIn(enforceSnippets),
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::Values(cpuParams_4D_planar)),
                        EltwiseCacheTest::getTestCaseName);

// shape for collapse test
std::vector<InputShapesTuple> inputShapes_4D_planar_collapse_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 25, 4, 5},
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 25, 1, 5},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 25, 4, 10},
                }
            },
            // in1.1
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 25, 1, 10},
                }
            }
        },
        // fq shapes
        {},
        // reshape shape
        {-1, 25, 4, 5}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_4D_planar_collapse_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_4D_planar_collapse_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(true), // needReshape
                                ::testing::ValuesIn(enforceSnippets),
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::Values(cpuParams_4D_planar)),
                        EltwiseCacheTest::getTestCaseName);

std::vector<InputShapesTuple> inputShapes_4D_blocked_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, {2, 64}, -1, -1},
                // target
                {
                    {2, 24, 40, 5},
                    {2, 24, 40, 5},
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, {2, 64}, -1, -1},
                // target
                {
                    {2, 24, 40, 5},
                    {2, 24, 40, 1},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, {2, 64}, -1, -1},
                // target
                {
                    {2, 48, 40, 10},
                    {2, 48, 40, 1},
                }
            },
            // in1.1
            {
                // dynamic
                {-1, {2, 64}, -1, -1},
                // target
                {
                    {2, 48, 40, 10},
                    {2, 48, 40, 10},
                }
            }
        },
        // fq shapes
        {},
        // reshape shape
        {-1, 24, 40, 5}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_4D_blocked_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_4D_blocked_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(true),
                                ::testing::Values(false),  // CPU Plugin doesn't support non-planar layout for Subgraphs
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_blocked_vec))),
                        EltwiseCacheTest::getTestCaseName);

std::vector<InputShapesTuple> inputShapes_4D_nspc_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 3, 4, 5},
                    {2, 3, 4, 5},
                    {2, 1, 4, 5},
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 3, 4, 5},
                    {2, 1, 4, 5},
                    {2, 3, 4, 5},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 6, 4, 10},
                    {2, 1, 4, 10},
                    {2, 1, 4, 10},
                }
            },
            // in1.1
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 6, 4, 10},
                    {2, 6, 4, 10},
                    {2, 12, 4, 10},
                }
            }
        },
        // fq shapes
        {},
        // reshape shape
        {-1, 3, 4, 5}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_4D_nspc_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_4D_nspc_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(true),
                                ::testing::Values(false),  // CPU Plugin doesn't support non-planar layout for Subgraphs
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(filterCPUSpecificParams({cpuParams_4D_nspc}))),
                        EltwiseCacheTest::getTestCaseName);

std::vector<InputShapesTuple> inputShapes_4D_nspc_collapse_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 3, 25, 5},
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 3, 25, 1},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 6, 25, 5},
                }
            },
            // in1.1
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {2, 6, 25, 1},
                }
            }
        },
        // fq shapes
        {},
        // reshape shape
        {-1, 3, 25, 5}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_4D_nspc_collapse_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_4D_nspc_collapse_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(true),
                                ::testing::Values(false),  // CPU Plugin doesn't support non-planar layout for Subgraphs
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(filterCPUSpecificParams({cpuParams_4D_nspc}))),
                        EltwiseCacheTest::getTestCaseName);


// 5D
CPUSpecificParams cpuParams_5D_planar = {{ncdhw, ncdhw}, {ncdhw}, {}, {}};
CPUSpecificParams cpuParams_5D_blocked = {{nCdhw16c, nCdhw16c}, {nCdhw16c}, {}, {}};
CPUSpecificParams cpuParams_5D_nspc = {{ndhwc, ndhwc}, {ndhwc}, {}, {}};
std::vector<CPUSpecificParams> cpuParams_5D_blocked_vec = {cpuParams_5D_blocked};

std::vector<InputShapesTuple> inputShapes_5D_planar_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 3, 4, 5, 6},
                    {2, 3, 4, 5, 6},
                    {2, 3, 4, 5, 1},
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 3, 4, 5, 6},
                    {2, 3, 4, 5, 1},
                    {2, 3, 4, 5, 6},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 6, 4, 5, 12},
                    {2, 6, 4, 5, 1},
                    {2, 6, 4, 5, 1},
                }
            },
            // in1.1
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 6, 4, 5, 12},
                    {2, 6, 4, 5, 12},
                    {2, 6, 4, 5, 24},
                }
            }
        },
        // fq shapes
        {
            {1, 3, 1, 1, 1}, {1, 6, 1, 1, 1}
        },
        // reshape shape
        {-1, 3, 4, 5, 6}
    }
};


INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_5D_planar_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_5D_planar_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false, true), // withQuantization
                                ::testing::Values(true), // needReshape
                                ::testing::ValuesIn(enforceSnippets),
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::Values(cpuParams_5D_planar)),
                        EltwiseCacheTest::getTestCaseName);

// shape for collapse test
std::vector<InputShapesTuple> inputShapes_5D_planar_collapse_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 25, 4, 5, 6},
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 25, 1, 5, 6},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 25, 4, 10, 6},
                }
            },
            // in1.1
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 25, 1, 10, 6},
                }
            }
        },
        // fq shapes
        {},
        // reshape shape
        {-1, 25, 4, 5, 6}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_5D_planar_collapse_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_5D_planar_collapse_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(true),
                                ::testing::ValuesIn(enforceSnippets),
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::Values(cpuParams_5D_planar)),
                        EltwiseCacheTest::getTestCaseName);

std::vector<InputShapesTuple> inputShapes_5D_blocked_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, {2, 64}, -1, -1, -1},
                // target
                {
                    {2, 24, 40, 5, 6},
                    {2, 24, 40, 5, 6},
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, {2, 64}, -1, -1, -1},
                // target
                {
                    {2, 24, 40, 5, 6},
                    {2, 24, 40, 5, 1},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, {2, 64}, -1, -1, -1},
                // target
                {
                    {2, 48, 40, 10, 12},
                    {2, 48, 40, 10, 1},
                }
            },
            // in1.1
            {
                // dynamic
                {-1, {2, 64}, -1, -1, -1},
                // target
                {
                    {2, 48, 40, 10, 12},
                    {2, 48, 40, 10, 12},
                }
            }
        },
        // fq shapes
        {},
        // reshape shape
        {-1, 24, 40, 5, 6}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_5D_blocked_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_5D_blocked_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(true),
                                ::testing::Values(false),  // CPU Plugin doesn't support non-planar layout for Subgraphs
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_blocked_vec))),
                        EltwiseCacheTest::getTestCaseName);

std::vector<InputShapesTuple> inputShapes_5D_nspc_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 3, 4, 5, 6},
                    {2, 3, 4, 5, 6},
                    {2, 1, 4, 5, 6},
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 3, 4, 5, 6},
                    {2, 1, 4, 5, 6},
                    {2, 3, 4, 5, 6},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 6, 4, 10, 12},
                    {2, 1, 4, 10, 12},
                    {2, 1, 4, 10, 12},
                }
            },
            // in1.1
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 6, 4, 10, 12},
                    {2, 6, 4, 10, 12},
                    {2, 12, 4, 10, 12},
                }
            }
        },
        // fq shapes
        {},
        // reshape shape
        {-1, 3, 4, 5, 6}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_5D_nspc_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_5D_nspc_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(true),
                                ::testing::Values(false),  // CPU Plugin doesn't support non-planar layout for Subgraphs
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(filterCPUSpecificParams({cpuParams_5D_nspc}))),
                        EltwiseCacheTest::getTestCaseName);

std::vector<InputShapesTuple> inputShapes_5D_nspc_collapse_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 3, 25, 5, 6},
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 3, 25, 1, 6},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 6, 25, 5, 6},
                }
            },
            // in1.1
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 6, 25, 1, 6},
                }
            }
        },
        // fq shapes
        {},
        // reshape shape
        {-1, 3, 25, 5, 6}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_5D_nspc_collapse_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_5D_nspc_collapse_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(true),
                                ::testing::Values(false),  // CPU Plugin doesn't support non-planar layout for Subgraphs
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(filterCPUSpecificParams({cpuParams_5D_nspc}))),
                        EltwiseCacheTest::getTestCaseName);


// 7D
std::vector<InputShapesTuple> inputShapes_7D_dyn = {
    {
        // eltwise shapes
        {
            // inp0.0
            {
                // dynamic
                {-1, -1, -1, -1, -1, -1, -1},
                // target
                {
                    {1, 2, 3, 4, 5, 6, 7},
                    {1, 2, 3, 4, 5, 6, 7},
                    {1, 2, 3, 4, 5, 6, 7},
                }
            },
            // inp0.1
            {
                // dynamic
                {-1, -1, -1, -1, -1, -1, -1},
                // target
                {
                    {1, 2, 3, 4, 5, 6, 7},
                    {1, 2, 3, 4, 5, 6, 1},
                    {1, 2, 3, 4, 5, 6, 7},
                }
            },
            // inp1.0
            {
                // dynamic
                {-1, -1, -1, -1, -1, -1, -1},
                // target
                {
                    {1, 4, 3, 4, 10, 6, 14},
                    {1, 4, 3, 4, 10, 6, 1},
                    {1, 4, 3, 4, 10, 6, 1},
                }
            },
            // in1.1
            {
                // dynamic
                {-1, -1, -1, -1, -1, -1, -1},
                // target
                {
                    {1, 4, 3, 4, 10, 6, 14},
                    {1, 4, 3, 4, 10, 6, 14},
                    {1, 4, 3, 4, 10, 6, 28},
                }
            }
        },
        // fq shapes
        {
            {1, 2, 1, 1, 1, 1, 1}, {1, 4, 1, 1, 1, 1, 1}
        },
        // reshape shape
        {-1, 2, 3, 4, 5, 6, 7}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseCache_7D_dyn, EltwiseCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_7D_dyn),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false, true), // withQuantization
                                ::testing::Values(true), // needReshape
                                ::testing::ValuesIn(enforceSnippets),
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::Values(cpuParams_empty)),
                        EltwiseCacheTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov