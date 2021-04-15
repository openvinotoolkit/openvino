// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cstdlib>
#include <gtest/gtest.h>

#include "test_utils.h"

#include "api/topology.hpp"
#include "api/network.hpp"
#include "api/input_layout.hpp"
#include "api/non_max_suppression.hpp"
#include "api/data.hpp"
#include "api/mutable_data.hpp"

using namespace cldnn;

template <typename T>
struct non_max_suppression_basic : public testing::Test {
    // Boxes:
    // batch 0:
    // 0. (0, 0) - (10, 10)
    // 1. (0, 2) - (9, 10) - iou 0: 0.72
    // 2. (5, 1) - (10, 10) - iou 0: 0.45, iou 1: 0.38
    // batch 1:
    // 0. (5, 0) - (10, 5)
    // 1. (0, 0) - (5, 5) - iou 0: 0
    // 2. (2, 0) - (9, 4) - iou 0: 0.43 iou 1: 0.29
    //
    // Scores:
    // batch.box  0.0    0.1    0.2    1.0    1.1    1.2
    // class
    // 0          0.3     0.7   0.9    0.25   0.5    0.8
    // 1          0.9     0.2   0.75   0.1    0.2    0.3
    //
    //
    // Sorted output:
    // batch  class  box  score
    //   0      0     2     0.9
    //   0      1     0     0.9
    //   1      0     2     0.8
    //   0      1     2     0.75   -- iou 0.45
    //   0      0     1     0.7    -- iou 0.38
    //   1      0     1     0.5    -- iou 0.29
    //   0      0     0     0.3    -- iou 0.72
    //   1      1     2     0.3
    //   1      0     0     0.25   -- iou 0.43
    //   0      1     1     0.2    -- iou 0.72
    //   1      1     1     0.2    -- iou 0.29
    //   1      1     0     0.1    -- iou 0.43
    const int batch_size = 2;
    const int classes_num = 2;
    const int boxes_num = 3;
    const int selected_indices_num = 6;

    const std::vector<T> boxes_data = {
        T(0.f), T(0.f), T(10.f), T(10.f),
        T(0.f), T(2.f), T(9.f), T(10.f),
        T(5.f), T(1.f), T(10.f), T(10.f),

        T(5.f), T(0.f), T(10.f), T(5.f),
        T(0.f), T(0.f), T(5.f), T(5.f),
        T(2.f), T(0.f), T(9.f), T(4.f),
    };

    const std::vector<T> scores_data = {
        T(0.3f), T(0.7f), T(0.9f),
        T(0.9f), T(0.2f), T(0.75f),
        T(0.25f), T(0.5f), T(0.8f),
        T(0.1f), T(0.2f), T(0.3f),
    };

    const layout boxes_layout = layout(type_to_data_type<T>::value, format::bfyx, tensor(batch(batch_size), feature(boxes_num), spatial(1, 4)));
    const layout scores_layout = layout(type_to_data_type<T>::value, format::bfyx, tensor(batch(batch_size), feature(classes_num), spatial(1, boxes_num)));

    const layout selected_scores_layout = layout(type_to_data_type<T>::value, format::bfyx, tensor(batch(selected_indices_num), feature(3)));
    const layout valid_outputs_layout = layout(cldnn::data_types::i32, format::bfyx, tensor(batch(selected_indices_num), feature(1)));

    memory get_boxes_memory(engine& engine) {
        auto mem = memory::allocate(engine, boxes_layout);
        tests::set_values(mem, boxes_data);
        return mem;
    }

    memory get_scores_memory(engine& engine) {
        auto mem = memory::allocate(engine, scores_layout);
        tests::set_values(mem, scores_data);
        return mem;
    }

    memory get_selected_scores_mem(engine& engine) {
        auto mem = memory::allocate(engine, selected_scores_layout);
        return mem;
    }

    memory get_valid_outputs_mem(engine& engine) {
        auto mem = memory::allocate(engine, valid_outputs_layout);
        return mem;
    }

    const int pad = -1;
};

using nms_types = testing::Types<float, half_t>;
TYPED_TEST_CASE(non_max_suppression_basic, nms_types);

TYPED_TEST(non_max_suppression_basic, basic) {
    auto engine = tests::get_test_engine();

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(non_max_suppression("nms", "boxes", "scores", 6, false, true));

    build_options build_opts(
        build_option::optimize_data(true)
    );
    auto net = network(engine, topo, build_opts);

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad
    };

    auto out_mem = result.at("nms").get_memory();
    auto out_ptr = out_mem.pointer<int>();

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}

TYPED_TEST(non_max_suppression_basic, optional_outputs) {
    auto engine = tests::get_test_engine();

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));

    memory selected_scores_mem = this->get_selected_scores_mem(engine);
    memory valid_outputs_mem = this->get_valid_outputs_mem(engine);

    topo.add(mutable_data("selected_scores", selected_scores_mem));
    topo.add(mutable_data("valid_outputs", valid_outputs_mem));

    topo.add(non_max_suppression("nms", "boxes", "scores", this->selected_indices_num, false, true,
                                cldnn::primitive_id(), cldnn::primitive_id(),
                                cldnn::primitive_id(), cldnn::primitive_id(),
                                "selected_scores", "valid_outputs"));

    build_options build_opts(
        build_option::optimize_data(true)
    );
    auto net = network(engine, topo, build_opts);

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad
    };

    auto out_mem = result.at("nms").get_memory();
    auto out_ptr = out_mem.pointer<int>();

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}

TYPED_TEST(non_max_suppression_basic, num_per_class) {
    auto engine = tests::get_test_engine();

    auto num_per_class_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(num_per_class_mem, { 1.f });

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(data("num_per_class", num_per_class_mem));
    topo.add(non_max_suppression("nms", "boxes", "scores", 
        this->batch_size * this->classes_num * 1, false, true, "num_per_class"));

    build_options build_opts(
        build_option::optimize_data(true)
    );
    auto net = network(engine, topo, build_opts);

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        0, 0, 2,
        0, 1, 0,
        1, 0, 2,
        1, 1, 2,
    };

    auto out_mem = result.at("nms").get_memory();
    auto out_ptr = out_mem.pointer<int>();

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}

TYPED_TEST(non_max_suppression_basic, iou_threshold) {
    auto engine = tests::get_test_engine();

    auto num_per_class_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(num_per_class_mem, { 3.f });
    auto iou_threshold_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(iou_threshold_mem, { 0.4f });

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(data("num_per_class", num_per_class_mem));
    topo.add(data("iou_threshold", iou_threshold_mem));
    topo.add(non_max_suppression("nms", "boxes", "scores", 
        this->batch_size * this->classes_num * this->boxes_num,
        false, true, "num_per_class", "iou_threshold"));

    build_options build_opts(
        build_option::optimize_data(true)
    );
    auto net = network(engine, topo, build_opts);

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        0, 0, 2,
        0, 1, 0,
        1, 0, 2,
        0, 0, 1,
        1, 0, 1,
        1, 1, 2,
        1, 1, 1,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad
    };

    auto out_mem = result.at("nms").get_memory();
    auto out_ptr = out_mem.pointer<int>();

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}

TYPED_TEST(non_max_suppression_basic, score_threshold) {
    auto engine = tests::get_test_engine();

    auto num_per_class_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(num_per_class_mem, { 3.f });
    auto iou_threshold_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(iou_threshold_mem, { 0.4f });
    auto score_threshold_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(score_threshold_mem, { 0.4f });

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(data("num_per_class", num_per_class_mem));
    topo.add(data("iou_threshold", iou_threshold_mem));
    topo.add(data("score_threshold", score_threshold_mem));
    topo.add(non_max_suppression("nms", "boxes", "scores",
        this->batch_size * this->classes_num * this->boxes_num,
        false, true, "num_per_class", "iou_threshold", "score_threshold"));

    build_options build_opts(
        build_option::optimize_data(true)
    );
    auto net = network(engine, topo, build_opts);

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        0, 0, 2,
        0, 1, 0,
        1, 0, 2,
        0, 0, 1,
        1, 0, 1,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad
    };

    auto out_mem = result.at("nms").get_memory();
    auto out_ptr = out_mem.pointer<int>();

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}

TYPED_TEST(non_max_suppression_basic, soft_nms_sigma) {
    // auto engine = tests::get_test_engine();

    cldnn::engine_configuration cfg = { true, false, false, "", "", true, "", "/home/sungeunk/work/openvino/build/cl_dump/"};
    auto engine = cldnn::engine(cfg); 

    auto num_per_class_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(num_per_class_mem, { 3.f });
    auto iou_threshold_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(iou_threshold_mem, { 0.4f });
    auto score_threshold_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(score_threshold_mem, { 0.4f });
    auto soft_nms_sigma_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(soft_nms_sigma_mem, { 0.5f });

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(data("num_per_class", num_per_class_mem));
    topo.add(data("iou_threshold", iou_threshold_mem));
    topo.add(data("score_threshold", score_threshold_mem));
    topo.add(data("soft_nms_sigma", soft_nms_sigma_mem));
    topo.add(non_max_suppression("nms", "boxes", "scores",
        this->batch_size * this->classes_num * this->boxes_num,
        false, true, "num_per_class", "iou_threshold", "score_threshold", "soft_nms_sigma"));

    build_options build_opts(
        build_option::optimize_data(true)
    );
    auto net = network(engine, topo, build_opts);

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    for(auto& p : net.get_executed_primitives()) {
        std::cout << p.first.c_str() << std::endl;
        for (auto pi : p.second.get_profiling_info()) {
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(pi.value->value()).count() / 1000.0;
            std::cout << "    " << pi.name << ": " << ms << " ms" << std::endl;
        }
    }

    std::vector<int> expected_out = {
        0, 0, 2,
        0, 1, 0,
        1, 0, 2,
        0, 0, 1,
        1, 0, 1,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad
    };

    auto out_mem = result.at("nms").get_memory();
    auto out_ptr = out_mem.pointer<int>();

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}



/*
 * Sort test
 */

typedef struct __Box {
    int batchId;
    int classId;
    int boxId;
    float score;
} Box;

typedef std::vector<std::vector<std::vector<Box>>>  Vector3Box;
typedef std::vector<std::vector<Box>>               Vector2Box;
typedef std::vector<Box>                            Vector1Box;

static bool compareBox (const Box& i, const Box& j) { return (i.score > j.score); }
static float generateScore() { return std::min(0.9999999f, std::max(0.0000001f, static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX))); }
static Vector3Box generateBoxInput(size_t batchNum, size_t classNum, size_t boxNum);
static std::vector<float> convertBoxToBoxCoord(Vector3Box& input);
static std::vector<float> convertBoxToBoxScore(Vector3Box& input);
static void sortBox(Vector3Box& input);
static bool compareOutput(Vector3Box& sortedBox, cldnn::pointer<int>& gpuResultBox);


template <typename T>
struct non_max_suppression_sort : public testing::Test {
    bool run(size_t batchNum, size_t classNum, size_t boxNum) {
        Vector3Box input = generateBoxInput(batchNum, classNum, boxNum);
        std::vector<float> boxes_data = convertBoxToBoxCoord(input);
        std::vector<float> scores_data = convertBoxToBoxScore(input);
        sortBox(input);

        auto engine = tests::get_test_engine();

        const layout boxes_layout = layout(type_to_data_type<float>::value, format::bfyx, tensor(batch(batchNum), feature(boxNum), spatial(1, 4)));
        const layout scores_layout = layout(type_to_data_type<float>::value, format::bfyx, tensor(batch(batchNum), feature(classNum), spatial(1, boxNum)));
        auto num_per_class_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
        tests::set_values(num_per_class_mem, { (float)boxNum });
        auto iou_threshold_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
        tests::set_values(iou_threshold_mem, { 1.1f });

        auto boxes_mem = memory::allocate(engine, boxes_layout);
        tests::set_values(boxes_mem, boxes_data);

        auto scores_mem = memory::allocate(engine, scores_layout);
        tests::set_values(scores_mem, scores_data);

        topology topo;
        topo.add(input_layout("boxes", boxes_layout));
        topo.add(input_layout("scores", scores_layout));
        topo.add(data("num_per_class", num_per_class_mem));
        topo.add(data("iou_threshold", iou_threshold_mem));
        topo.add(non_max_suppression("nms", "boxes", "scores", 
            batchNum * classNum * boxNum, false, false, "num_per_class", "iou_threshold"));

        build_options build_opts(
            build_option::optimize_data(true)
        );
        auto net = network(engine, topo, build_opts);

        net.set_input_data("boxes", boxes_mem);
        net.set_input_data("scores", scores_mem);

        auto result = net.execute();
        auto out_mem = result.at("nms").get_memory();
        auto out_ptr = out_mem.pointer<int>();

        return compareOutput(input, out_ptr);
    }
};

TYPED_TEST_CASE(non_max_suppression_sort, nms_types);

TYPED_TEST(non_max_suppression_sort, sort_1_80_400) {
    auto ret = this->run(1, 80, 400);
    EXPECT_TRUE(ret);
}

Vector3Box generateBoxInput(size_t batchNum, size_t classNum, size_t boxNum) {
    Vector3Box boxVec(batchNum, Vector2Box(classNum, Vector1Box(boxNum)));
    for (size_t batchId = 0; batchId < batchNum; batchId++) {
        for (size_t classId = 0; classId < classNum; classId++) {
            for (size_t boxId = 0; boxId < boxNum; boxId++) {
                boxVec[batchId][classId][boxId] = { (int)batchId, (int)classId, (int)boxId, generateScore() };
            }
        }
    }

    return boxVec;
}

// Never mind the box coords.
std::vector<float> convertBoxToBoxCoord(Vector3Box& input) {
    size_t batchNum = input.size();
    size_t boxNum = input[0][0].size();
    std::vector<float> coordVec(batchNum * boxNum * 4);
    return coordVec;
}

std::vector<float> convertBoxToBoxScore(Vector3Box& input) {
    size_t batchNum = input.size();
    size_t classNum = input[0].size();
    size_t boxNum = input[0][0].size();
    const size_t totalBoxNum = batchNum * classNum * boxNum;
    std::vector<float> scoreVec(totalBoxNum);

    size_t i = 0;
    for (size_t batchId = 0; batchId < batchNum; batchId++) {
        for (size_t classId = 0; classId < classNum; classId++) {
            for (size_t boxId = 0; boxId < boxNum; boxId++) {
                scoreVec[i++] = input[batchId][classId][boxId].score;
            }
        }
    }
    return scoreVec;
}

void sortBox(Vector3Box& input) {
    size_t batchNum = input.size();
    size_t classNum = input[0].size();

    for (size_t batchId = 0; batchId < batchNum; batchId++) {
        for (size_t classId = 0; classId < classNum; classId++) {
            std::sort(input[batchId][classId].begin(), input[batchId][classId].end(), compareBox);
        }
    }
}

bool compareOutput(Vector3Box& sortedBox, cldnn::pointer<int>& gpuResultBox) {
    size_t batchNum = sortedBox.size();
    size_t classNum = sortedBox[0].size();
    size_t boxNum = sortedBox[0][0].size();

    bool ret = true;
    size_t i = 0;
    for (size_t batchId = 0; batchId < batchNum; batchId++) {
        for (size_t classId = 0; classId < classNum; classId++) {
            for (size_t boxId = 0; boxId < boxNum; boxId++, i+=3) {
                Box& box = sortedBox[batchId][classId][boxId];
                if (box.batchId != gpuResultBox[i] || 
                    box.classId != gpuResultBox[i+1] ||
                    box.boxId != gpuResultBox[i+2]) {
                    printf("Box(%d/%d/%d+%.2f) result(%d/%d/%d)\n", box.batchId, box.classId, box.boxId, box.score, gpuResultBox[i], gpuResultBox[i+1], gpuResultBox[i+2]);
                    ret = false;
                }
            }
        }
    }

    return ret;
}
