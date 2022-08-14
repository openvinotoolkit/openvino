// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/primitives/non_max_suppression.hpp>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

template <typename DataType, cldnn::format::type l>
struct TypeWithLayoutFormat {
    using Type = DataType;
    static const cldnn::format::type layout = l;
};

template <typename TypeWithLayout>
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
    using DataType = typename TypeWithLayout::Type;
    const int batch_size = 2;
    const int classes_num = 2;
    const int boxes_num = 3;
    const int selected_indices_num = 6;

    const std::vector<DataType> boxes_data = {
        DataType(0.f), DataType(0.f),  DataType(10.f), DataType(10.f), DataType(0.f),  DataType(2.f),
        DataType(9.f), DataType(10.f), DataType(5.f),  DataType(1.f),  DataType(10.f), DataType(10.f),
        DataType(5.f), DataType(0.f),  DataType(10.f), DataType(5.f),  DataType(0.f),  DataType(0.f),
        DataType(5.f), DataType(5.f),  DataType(2.f),  DataType(0.f),  DataType(9.f),  DataType(4.f),
    };

    const std::vector<DataType> scores_data = {
        DataType(0.3f),
        DataType(0.7f),
        DataType(0.9f),
        DataType(0.9f),
        DataType(0.2f),
        DataType(0.75f),
        DataType(0.25f),
        DataType(0.5f),
        DataType(0.8f),
        DataType(0.1f),
        DataType(0.2f),
        DataType(0.3f),
    };

    const layout boxes_layout = layout(type_to_data_type<DataType>::value,
                                       format::bfyx,
                                       tensor(batch(batch_size), feature(boxes_num), spatial(1, 4)));
    const layout scores_layout = layout(type_to_data_type<DataType>::value,
                                        format::bfyx,
                                        tensor(batch(batch_size), feature(classes_num), spatial(1, boxes_num)));
    const layout selected_scores_layout =
        layout(data_type, layout_format, tensor(batch(selected_indices_num), feature(3)));
    const layout valid_outputs_layout = layout(cldnn::data_types::i32, layout_format, tensor(batch(1)));

    memory::ptr get_boxes_memory(engine& engine) {
        auto mem = engine.allocate_memory(boxes_layout);
        tests::set_values(mem, boxes_data);
        return mem;
    }

    memory::ptr get_scores_memory(engine& engine) {
        auto mem = engine.allocate_memory(scores_layout);
        tests::set_values(mem, scores_data);
        return mem;
    }

    memory::ptr get_selected_scores_mem(engine& engine) {
        auto mem = engine.allocate_memory(selected_scores_layout);
        return mem;
    }

    memory::ptr get_valid_outputs_mem(engine& engine) {
        auto mem = engine.allocate_memory(valid_outputs_layout);
        return mem;
    }

    static const format::type layout_format = TypeWithLayout::layout;

    static const data_types data_type = type_to_data_type<DataType>::value;

    const int pad = -1;
};

using nms_types = testing::Types<TypeWithLayoutFormat<float, cldnn::format::bfyx>,
                                 TypeWithLayoutFormat<float, cldnn::format::b_fs_yx_fsv32>,
                                 TypeWithLayoutFormat<float, cldnn::format::b_fs_yx_fsv16>,
                                 TypeWithLayoutFormat<float, cldnn::format::bs_fs_yx_bsv32_fsv16>,
                                 TypeWithLayoutFormat<float, cldnn::format::bs_fs_yx_bsv16_fsv16>,
                                 TypeWithLayoutFormat<float, cldnn::format::bs_fs_yx_bsv32_fsv32>,

                                 TypeWithLayoutFormat<half_t, cldnn::format::bfyx>,
                                 TypeWithLayoutFormat<half_t, cldnn::format::b_fs_yx_fsv32>,
                                 TypeWithLayoutFormat<half_t, cldnn::format::b_fs_yx_fsv16>,
                                 TypeWithLayoutFormat<half_t, cldnn::format::bs_fs_yx_bsv32_fsv16>,
                                 TypeWithLayoutFormat<half_t, cldnn::format::bs_fs_yx_bsv16_fsv16>,
                                 TypeWithLayoutFormat<half_t, cldnn::format::bs_fs_yx_bsv32_fsv32>>;
TYPED_TEST_SUITE(non_max_suppression_basic, nms_types);

TYPED_TEST(non_max_suppression_basic, basic) {
    auto& engine = tests::get_test_engine();

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(reorder("reformat_boxes", "boxes", this->layout_format, this->data_type));
    topo.add(reorder("reformat_scores", "scores", this->layout_format, this->data_type));
    topo.add(non_max_suppression("nms", "reformat_boxes", "reformat_scores", 6, false, true));
    topo.add(reorder("plane_nms", "nms", format::bfyx, cldnn::data_types::i32));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    cldnn::network net{engine, topo, bo};

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad,
                                     this->pad};

    auto out_mem = result.at("plane_nms").get_memory();
    cldnn::mem_lock<int> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}

TYPED_TEST(non_max_suppression_basic, num_per_class) {
    auto& engine = tests::get_test_engine();
    auto num_per_class_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(num_per_class_mem, {1.f});

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(data("num_per_class", num_per_class_mem));
    topo.add(reorder("reformat_boxes", "boxes", this->layout_format, this->data_type),
             reorder("reformat_scores", "scores", this->layout_format, this->data_type),
             non_max_suppression("nms",
                                 "reformat_boxes",
                                 "reformat_scores",
                                 this->batch_size * this->classes_num * 1,
                                 false,
                                 true,
                                 "num_per_class"));
    topo.add(reorder("plane_nms", "nms", format::bfyx, cldnn::data_types::i32));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    cldnn::network net{engine, topo, bo};

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        0,
        0,
        2,
        0,
        1,
        0,
        1,
        0,
        2,
        1,
        1,
        2,
    };

    auto out_mem = result.at("plane_nms").get_memory();
    cldnn::mem_lock<int> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}

TYPED_TEST(non_max_suppression_basic, optional_outputs) {
    auto& engine = tests::get_test_engine();

    auto num_per_class_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(num_per_class_mem, {1.f});

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(data("num_per_class", num_per_class_mem));

    memory::ptr selected_scores_mem = this->get_selected_scores_mem(engine);
    memory::ptr valid_outputs_mem = this->get_valid_outputs_mem(engine);

    topo.add(mutable_data("selected_scores", selected_scores_mem));
    topo.add(mutable_data("valid_outputs", valid_outputs_mem));

    topo.add(reorder("reformat_boxes", "boxes", this->layout_format, this->data_type),
             reorder("reformat_scores", "scores", this->layout_format, this->data_type),
             non_max_suppression("nms",
                                 "reformat_boxes",
                                 "reformat_scores",
                                 this->batch_size * this->classes_num * 1,
                                 false,
                                 true,
                                 "num_per_class",
                                 cldnn::primitive_id(),
                                 cldnn::primitive_id(),
                                 cldnn::primitive_id(),
                                 "selected_scores",
                                 "valid_outputs"));
    topo.add(reorder("plane_nms", "nms", format::bfyx, cldnn::data_types::i32));
    topo.add(reorder("plane_scores", "selected_scores", format::bfyx, this->data_type));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    cldnn::network net{engine, topo, bo};

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        0,
        0,
        2,
        0,
        1,
        0,
        1,
        0,
        2,
        1,
        1,
        2,
    };
    const int expected_out_num = static_cast<int>(expected_out.size()) / 3;

    std::vector<float> expected_second_out = {
        0.f,
        0.f,
        0.9f,
        0.f,
        1.f,
        0.9f,
        1.f,
        0.f,
        0.8f,
        1.f,
        1.f,
        0.3f,
    };

    auto out_mem = result.at("plane_nms").get_memory();
    cldnn::mem_lock<int> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }

    topology second_output_topology;
    second_output_topology.add(input_layout("selected_scores", this->selected_scores_layout));
    second_output_topology.add(input_layout("num_outputs", this->valid_outputs_layout));
    second_output_topology.add(reorder("plane_scores", "selected_scores", format::bfyx, this->data_type));
    second_output_topology.add(reorder("plane_num", "num_outputs", format::bfyx, cldnn::data_types::i32));
    network second_output_net{engine, second_output_topology};
    second_output_net.set_input_data("selected_scores", selected_scores_mem);
    second_output_net.set_input_data("num_outputs", valid_outputs_mem);
    auto second_output_result = second_output_net.execute();
    auto plane_scores_mem = second_output_result.at("plane_scores").get_memory();
    if (this->data_type == data_types::f32) {
        cldnn::mem_lock<float> second_output_ptr(plane_scores_mem, get_test_stream());

        for (size_t i = 0; i < expected_second_out.size(); ++i) {
            EXPECT_FLOAT_EQ(expected_second_out[i], second_output_ptr[i]);
        }
    } else {
        cldnn::mem_lock<half_t> second_output_ptr(plane_scores_mem, get_test_stream());

        for (size_t i = 0; i < expected_second_out.size(); ++i) {
            EXPECT_NEAR(expected_second_out[i], half_to_float(second_output_ptr[i]), 0.0002f);
        }
    }

    cldnn::mem_lock<int> third_output_ptr(second_output_result.at("plane_num").get_memory(), get_test_stream());
    ASSERT_EQ(expected_out_num, third_output_ptr[0]);
}

TYPED_TEST(non_max_suppression_basic, iou_threshold) {
    auto& engine = tests::get_test_engine();

    auto num_per_class_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(num_per_class_mem, {3.f});
    auto iou_threshold_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(iou_threshold_mem, {0.4f});

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(data("num_per_class", num_per_class_mem));
    topo.add(data("iou_threshold", iou_threshold_mem));
    topo.add(reorder("reformat_boxes", "boxes", this->layout_format, this->data_type),
             reorder("reformat_scores", "scores", this->layout_format, this->data_type),
             non_max_suppression("nms",
                                 "reformat_boxes",
                                 "reformat_scores",
                                 this->batch_size * this->classes_num * this->boxes_num,
                                 false,
                                 true,
                                 "num_per_class",
                                 "iou_threshold"));
    topo.add(reorder("plane_nms", "nms", format::bfyx, cldnn::data_types::i32));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    cldnn::network net{engine, topo, bo};

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        0,         0,         2,         0,         1,         0,         1,         0,         2,
        0,         0,         1,         1,         0,         1,         1,         1,         2,
        1,         1,         1,         this->pad, this->pad, this->pad, this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad};

    auto out_mem = result.at("plane_nms").get_memory();
    cldnn::mem_lock<int> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}

TYPED_TEST(non_max_suppression_basic, score_threshold) {
    auto& engine = tests::get_test_engine();

    auto num_per_class_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(num_per_class_mem, {3.f});
    auto iou_threshold_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(iou_threshold_mem, {0.4f});
    auto score_threshold_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(score_threshold_mem, {0.4f});

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(data("num_per_class", num_per_class_mem));
    topo.add(data("iou_threshold", iou_threshold_mem));
    topo.add(data("score_threshold", score_threshold_mem));
    topo.add(reorder("reformat_boxes", "boxes", this->layout_format, this->data_type),
             reorder("reformat_scores", "scores", this->layout_format, this->data_type),
             non_max_suppression("nms",
                                 "reformat_boxes",
                                 "reformat_scores",
                                 this->batch_size * this->classes_num * this->boxes_num,
                                 false,
                                 true,
                                 "num_per_class",
                                 "iou_threshold",
                                 "score_threshold"));
    topo.add(reorder("plane_nms", "nms", format::bfyx, cldnn::data_types::i32));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    cldnn::network net{engine, topo, bo};

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        0,         0,         2,         0,         1,         0,         1,         0,         2,
        0,         0,         1,         1,         0,         1,         this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad};

    auto out_mem = result.at("plane_nms").get_memory();
    cldnn::mem_lock<int> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}

TYPED_TEST(non_max_suppression_basic, soft_nms_sigma) {
    auto& engine = tests::get_test_engine();

    auto num_per_class_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(num_per_class_mem, {3.f});
    auto iou_threshold_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(iou_threshold_mem, {0.4f});
    auto score_threshold_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(score_threshold_mem, {0.4f});
    auto soft_nms_sigma_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(soft_nms_sigma_mem, {0.5f});

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(data("num_per_class", num_per_class_mem));
    topo.add(data("iou_threshold", iou_threshold_mem));
    topo.add(data("score_threshold", score_threshold_mem));
    topo.add(data("soft_nms_sigma", soft_nms_sigma_mem));
    topo.add(reorder("reformat_boxes", "boxes", this->layout_format, this->data_type),
             reorder("reformat_scores", "scores", this->layout_format, this->data_type),
             non_max_suppression("nms",
                                 "reformat_boxes",
                                 "reformat_scores",
                                 this->batch_size * this->classes_num * this->boxes_num,
                                 false,
                                 true,
                                 "num_per_class",
                                 "iou_threshold",
                                 "score_threshold",
                                 "soft_nms_sigma"));
    topo.add(reorder("plane_nms", "nms", format::bfyx, cldnn::data_types::i32));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    cldnn::network net{engine, topo, bo};

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        0,         0,         2,         0,         1,         0,         1,         0,         2,
        0,         1,         2,         0,         0,         1,         1,         0,         1,
        this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad};

    auto out_mem = result.at("plane_nms").get_memory();
    cldnn::mem_lock<int> out_ptr(out_mem, get_test_stream());
    std::vector<int64_t> score_indices;
    score_indices.resize(36);
    std::vector<float> sel_scores(36);
    std::vector<int64_t> outp;
    outp.resize(36);
    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}
