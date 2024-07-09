// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
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
    static const format::type layout_format = TypeWithLayout::layout;
    const data_types data_type = ov::element::from<DataType>();

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

    const layout boxes_layout = layout(ov::PartialShape{batch_size, boxes_num, 4},
                                       data_type,
                                       format::bfyx);
    const layout scores_layout = layout(ov::PartialShape{batch_size, classes_num, boxes_num},
                                        data_type,
                                        format::bfyx);

    const layout selected_scores_layout = layout(ov::PartialShape{selected_indices_num, 3}, data_type, layout_format);
    const layout valid_outputs_layout = layout(ov::PartialShape{1}, cldnn::data_types::i32, layout_format);

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


    const int pad = -1;

    void test_basic(bool is_caching_test) {
        auto& engine = tests::get_test_engine();

        topology topo;
        topo.add(input_layout("boxes", this->boxes_layout));
        topo.add(input_layout("scores", this->scores_layout));
        topo.add(reorder("reformat_boxes", input_info("boxes"), this->layout_format, this->data_type));
        topo.add(reorder("reformat_scores", input_info("scores"), this->layout_format, this->data_type));
        topo.add(non_max_suppression("nms", input_info("reformat_boxes"), input_info("reformat_scores"), 6, false, true));
        topo.add(reorder("plane_nms", input_info("nms"), format::bfyx, cldnn::data_types::i32));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        auto boxes_mem = this->get_boxes_memory(engine);
        auto scores_mem = this->get_scores_memory(engine);

        net->set_input_data("boxes", boxes_mem);
        net->set_input_data("scores", scores_mem);

        auto result = net->execute();

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
            ASSERT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
        }
    }

    void test_num_per_class(bool is_caching_test) {
        auto& engine = tests::get_test_engine();
        auto num_per_class_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
        tests::set_values(num_per_class_mem, {1.f});

        topology topo;
        topo.add(input_layout("boxes", this->boxes_layout));
        topo.add(input_layout("scores", this->scores_layout));
        topo.add(data("num_per_class", num_per_class_mem));
        topo.add(reorder("reformat_boxes", input_info("boxes"), this->layout_format, this->data_type),
                reorder("reformat_scores", input_info("scores"), this->layout_format, this->data_type),
                non_max_suppression("nms",
                                    input_info("reformat_boxes"),
                                    input_info("reformat_scores"),
                                    this->batch_size * this->classes_num * 1,
                                    false,
                                    true,
                                    "num_per_class"));
        topo.add(reorder("plane_nms", input_info("nms"), format::bfyx, cldnn::data_types::i32));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        auto boxes_mem = this->get_boxes_memory(engine);
        auto scores_mem = this->get_scores_memory(engine);

        net->set_input_data("boxes", boxes_mem);
        net->set_input_data("scores", scores_mem);

        auto result = net->execute();

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
            ASSERT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
        }
    }

    void test_optional_outputs(bool is_caching_test) {
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

        topo.add(reorder("reformat_boxes", input_info("boxes"), this->layout_format, this->data_type),
                reorder("reformat_scores", input_info("scores"), this->layout_format, this->data_type),
                non_max_suppression("nms",
                                    input_info("reformat_boxes"),
                                    input_info("reformat_scores"),
                                    this->batch_size * this->classes_num * 1,
                                    false,
                                    true,
                                    "num_per_class",
                                    cldnn::primitive_id(),
                                    cldnn::primitive_id(),
                                    cldnn::primitive_id(),
                                    "selected_scores",
                                    "valid_outputs"));
        topo.add(reorder("plane_nms", input_info("nms"), format::bfyx, cldnn::data_types::i32));
        topo.add(reorder("plane_scores", input_info("selected_scores"), format::bfyx, this->data_type));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        auto boxes_mem = this->get_boxes_memory(engine);
        auto scores_mem = this->get_scores_memory(engine);

        net->set_input_data("boxes", boxes_mem);
        net->set_input_data("scores", scores_mem);

        auto result = net->execute();

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
            ASSERT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
        }

        if (is_caching_test)
            return;

        topology second_output_topology;
        second_output_topology.add(input_layout("selected_scores", this->selected_scores_layout));
        second_output_topology.add(input_layout("num_outputs", this->valid_outputs_layout));
        second_output_topology.add(reorder("plane_scores", input_info("selected_scores"), format::bfyx, this->data_type));
        second_output_topology.add(reorder("plane_num", input_info("num_outputs"), format::bfyx, cldnn::data_types::i32));
        network second_output_net{engine, second_output_topology, get_test_default_config(engine)};
        second_output_net.set_input_data("selected_scores", selected_scores_mem);
        second_output_net.set_input_data("num_outputs", valid_outputs_mem);
        auto second_output_result = second_output_net.execute();
        auto plane_scores_mem = second_output_result.at("plane_scores").get_memory();
        if (this->data_type == data_types::f32) {
            cldnn::mem_lock<float> second_output_ptr(plane_scores_mem, get_test_stream());

            for (size_t i = 0; i < expected_second_out.size(); ++i) {
                ASSERT_FLOAT_EQ(expected_second_out[i], second_output_ptr[i]);
            }
        } else {
            cldnn::mem_lock<ov::float16> second_output_ptr(plane_scores_mem, get_test_stream());

            for (size_t i = 0; i < expected_second_out.size(); ++i) {
                ASSERT_NEAR(expected_second_out[i], static_cast<float>(second_output_ptr[i]), 0.0002f);
            }
        }

        cldnn::mem_lock<int> third_output_ptr(second_output_result.at("plane_num").get_memory(), get_test_stream());
        ASSERT_EQ(expected_out_num, third_output_ptr[0]);
    }

    void test_multiple_outputs(bool is_caching_test) {
        auto& engine = tests::get_test_engine();

        auto num_per_class_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
        tests::set_values(num_per_class_mem, {1.f});

        topology topo;
        const auto l_boxes = this->boxes_layout;
        topo.add(input_layout("boxes", layout{ov::PartialShape{l_boxes.batch(), l_boxes.feature(), l_boxes.spatial(1)}, l_boxes.data_type, l_boxes.format}));
        const auto l_scores = this->scores_layout;
        topo.add(input_layout("scores", layout{ov::PartialShape{l_scores.batch(), l_scores.feature(), l_scores.spatial(1)}, l_scores.data_type, l_scores.format}));
        topo.add(data("num_per_class", num_per_class_mem));
        topo.add(reorder("reformat_boxes", input_info("boxes"), this->layout_format, this->data_type),
                reorder("reformat_scores", input_info("scores"), this->layout_format, this->data_type));
        auto nms = non_max_suppression("nms",
                                    input_info("reformat_boxes"),
                                    input_info("reformat_scores"),
                                    this->batch_size * this->classes_num * 1,
                                    false,
                                    true,
                                    "num_per_class",
                                    cldnn::primitive_id(),
                                    cldnn::primitive_id(),
                                    cldnn::primitive_id(),
                                    cldnn::primitive_id(),
                                    cldnn::primitive_id(),
                                    3);
        auto output_data_type = this->data_type;
        nms.output_data_types = {optional_data_type{}, optional_data_type{output_data_type}, optional_data_type{}};
        nms.output_paddings = {padding(), padding(), padding()};
        topo.add(nms);
        topo.add(reorder("plane_nms", input_info("nms", 0), format::bfyx, cldnn::data_types::i32));
        topo.add(reorder("plane_scores", input_info("nms", 1), format::bfyx, this->data_type));
        topo.add(reorder("plane_outputs", input_info("nms", 2), format::bfyx, cldnn::data_types::i32));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        auto boxes_mem = this->get_boxes_memory(engine);
        auto scores_mem = this->get_scores_memory(engine);

        net->set_input_data("boxes", boxes_mem);
        net->set_input_data("scores", scores_mem);

        auto result = net->execute();

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

        auto selected_scores_mem = result.at("plane_scores").get_memory();
        auto valid_outputs_mem = result.at("plane_outputs").get_memory();

        ASSERT_EQ(expected_out.size(), out_ptr.size());
        for (size_t i = 0; i < expected_out.size(); ++i) {
            ASSERT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
        }


        topology second_output_topology;
        second_output_topology.add(input_layout("selected_scores", selected_scores_mem->get_layout()));
        second_output_topology.add(input_layout("num_outputs", valid_outputs_mem->get_layout()));
        second_output_topology.add(reorder("plane_scores", input_info("selected_scores"), format::bfyx, this->data_type));
        second_output_topology.add(reorder("plane_num", input_info("num_outputs"), format::bfyx, cldnn::data_types::i32));
        network second_output_net{engine, second_output_topology, get_test_default_config(engine)};
        second_output_net.set_input_data("selected_scores", selected_scores_mem);
        second_output_net.set_input_data("num_outputs", valid_outputs_mem);
        auto second_output_result = second_output_net.execute();
        auto plane_scores_mem = second_output_result.at("plane_scores").get_memory();
        if (this->data_type == data_types::f32) {
            cldnn::mem_lock<float> second_output_ptr(plane_scores_mem, get_test_stream());

            for (size_t i = 0; i < expected_second_out.size(); ++i) {
                ASSERT_FLOAT_EQ(expected_second_out[i], second_output_ptr[i]);
            }
        } else {
            cldnn::mem_lock<ov::float16> second_output_ptr(plane_scores_mem, get_test_stream());

            for (size_t i = 0; i < expected_second_out.size(); ++i) {
                ASSERT_NEAR(expected_second_out[i], static_cast<float>(second_output_ptr[i]), 0.0002f);
            }
        }

        cldnn::mem_lock<int> third_output_ptr(second_output_result.at("plane_num").get_memory(), get_test_stream());
        ASSERT_EQ(expected_out_num, third_output_ptr[0]);
    }

    void test_iou_threshold(bool is_caching_test) {
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
        topo.add(reorder("reformat_boxes", input_info("boxes"), this->layout_format, this->data_type),
                reorder("reformat_scores", input_info("scores"), this->layout_format, this->data_type),
                non_max_suppression("nms",
                                    input_info("reformat_boxes"),
                                    input_info("reformat_scores"),
                                    this->batch_size * this->classes_num * this->boxes_num,
                                    false,
                                    true,
                                    "num_per_class",
                                    "iou_threshold"));
        topo.add(reorder("plane_nms", input_info("nms"), format::bfyx, cldnn::data_types::i32));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        auto boxes_mem = this->get_boxes_memory(engine);
        auto scores_mem = this->get_scores_memory(engine);

        net->set_input_data("boxes", boxes_mem);
        net->set_input_data("scores", scores_mem);

        auto result = net->execute();

        std::vector<int> expected_out = {
            0,         0,         2,         0,         1,         0,         1,         0,         2,
            0,         0,         1,         1,         0,         1,         1,         1,         2,
            1,         1,         1,         this->pad, this->pad, this->pad, this->pad, this->pad, this->pad,
            this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad};

        auto out_mem = result.at("plane_nms").get_memory();
        cldnn::mem_lock<int> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(expected_out.size(), out_ptr.size());
        for (size_t i = 0; i < expected_out.size(); ++i) {
            ASSERT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
        }
    }

    void test_score_threshold(bool is_caching_test) {
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
        topo.add(reorder("reformat_boxes", input_info("boxes"), this->layout_format, this->data_type),
                reorder("reformat_scores", input_info("scores"), this->layout_format, this->data_type),
                non_max_suppression("nms",
                                    input_info("reformat_boxes"),
                                    input_info("reformat_scores"),
                                    this->batch_size * this->classes_num * this->boxes_num,
                                    false,
                                    true,
                                    "num_per_class",
                                    "iou_threshold",
                                    "score_threshold"));
        topo.add(reorder("plane_nms", input_info("nms"), format::bfyx, cldnn::data_types::i32));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        auto boxes_mem = this->get_boxes_memory(engine);
        auto scores_mem = this->get_scores_memory(engine);

        net->set_input_data("boxes", boxes_mem);
        net->set_input_data("scores", scores_mem);

        auto result = net->execute();

        std::vector<int> expected_out = {
            0,         0,         2,         0,         1,         0,         1,         0,         2,
            0,         0,         1,         1,         0,         1,         this->pad, this->pad, this->pad,
            this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad,
            this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad, this->pad};

        auto out_mem = result.at("plane_nms").get_memory();
        cldnn::mem_lock<int> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(expected_out.size(), out_ptr.size());
        for (size_t i = 0; i < expected_out.size(); ++i) {
            ASSERT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
        }
    }

    void test_nms_gather_score_threshold(bool is_caching_test) {
        auto& engine = tests::get_test_engine();

        auto num_per_class_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
        tests::set_values(num_per_class_mem, {3.f});
        auto iou_threshold_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
        tests::set_values(iou_threshold_mem, {0.4f});
        auto score_threshold_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
        tests::set_values(score_threshold_mem, {0.4f});

        const auto l_boxes = this->boxes_layout;
        const auto l_scores = this->scores_layout;

        topology topo;
        topo.add(input_layout("boxes", layout{ov::PartialShape{l_boxes.batch(), l_boxes.feature(), l_boxes.spatial(1)}, l_boxes.data_type, l_boxes.format}));
        topo.add(input_layout("scores", layout{ov::PartialShape{l_scores.batch(), l_scores.feature(), l_scores.spatial(1)}, l_scores.data_type, l_scores.format}));
        topo.add(data("num_per_class", num_per_class_mem));
        topo.add(data("iou_threshold", iou_threshold_mem));
        topo.add(data("score_threshold", score_threshold_mem));
        topo.add(reorder("reformat_boxes", input_info("boxes"), this->layout_format, this->data_type));
        topo.add(reorder("reformat_scores", input_info("scores"), this->layout_format, this->data_type));

        auto nms = non_max_suppression("nms",
                                    input_info("reformat_boxes"),
                                    input_info("reformat_scores"),
                                    this->batch_size * this->classes_num * this->boxes_num,
                                    false,
                                    true,
                                    "num_per_class",
                                    "iou_threshold",
                                    "score_threshold",
                                    "", "", "", 3);
        auto output_data_type = this->data_type;
        nms.output_data_types = {optional_data_type{}, optional_data_type{output_data_type}, optional_data_type{}};
        nms.output_paddings = {padding(), padding(), padding()};
        
        topo.add(nms);
        topo.add(non_max_suppression_gather("nms_gather",
                                            {input_info("nms", 0),
                                            input_info("nms", 1),
                                            input_info("nms", 2)},
                                            3));
        topo.add(reorder("plane_nms0", input_info("nms_gather", 0), format::bfyx, cldnn::data_types::i32));
        topo.add(reorder("plane_nms1", input_info("nms_gather", 1), format::bfyx, this->data_type));
        topo.add(reorder("plane_nms2", input_info("nms_gather", 2), format::bfyx, cldnn::data_types::i32));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        auto boxes_mem = this->get_boxes_memory(engine);
        auto scores_mem = this->get_scores_memory(engine);

        net->set_input_data("boxes", boxes_mem);
        net->set_input_data("scores", scores_mem);

        auto result = net->execute();

        // output 0
        std::vector<int> expected_out0 = {
            0, 0, 2,
            0, 1, 0,
            1, 0, 2,
            0, 0, 1,
            1, 0, 1
        };

        auto out_mem0 = result.at("plane_nms0").get_memory();
        cldnn::mem_lock<int> out0_ptr(out_mem0, get_test_stream());

        ASSERT_EQ(expected_out0.size(), out0_ptr.size());
        for (size_t i = 0; i < out0_ptr.size(); ++i) {
            ASSERT_EQ(expected_out0[i], out0_ptr[i]) << "at i = " << i;
        }

        // output 1
        if (this->data_type == cldnn::data_types::f32) {
            std::vector<float> expected_out1 = {
                0.0f, 0.0f, 0.9f,
                0.0f, 1.0f, 0.9f,
                1.0f, 0.0f, 0.8f,
                0.0f, 0.0f, 0.7f,
                1.0f, 0.0f, 0.5f
            };
            auto out_mem1 = result.at("plane_nms1").get_memory();
            cldnn::mem_lock<float> out1_ptr(out_mem1, get_test_stream());

            ASSERT_EQ(expected_out1.size(), out1_ptr.size());
            for (size_t i = 0; i < out1_ptr.size(); ++i) {
                ASSERT_EQ(expected_out1[i], out1_ptr[i]) << "at i = " << i;
            }
        } else if (this->data_type == cldnn::data_types::f16) {
            std::vector<ov::float16> expected_out1 = {
                0.0f, 0.0f, 0.899902f,
                0.0f, 1.0f, 0.899902f,
                1.0f, 0.0f, 0.799805f,
                0.0f, 0.0f, 0.700195f,
                1.0f, 0.0f, 0.5f
            };
            auto out_mem1 = result.at("plane_nms1").get_memory();
            cldnn::mem_lock<ov::float16> out1_ptr(out_mem1, get_test_stream());

            ASSERT_EQ(expected_out1.size(), out1_ptr.size());
            for (size_t i = 0; i < out1_ptr.size(); ++i) {
                ASSERT_EQ(expected_out1[i], out1_ptr[i]) << "at i = " << i;
            }
        } else {
            GTEST_FAIL() << "Not supported data type.";
        }

        // output 2
        auto out_mem2 = result.at("plane_nms2").get_memory();
        cldnn::mem_lock<int> out2_ptr(out_mem2, get_test_stream());
        ASSERT_EQ(1, out2_ptr.size());
        ASSERT_EQ(5, out2_ptr[0]);
    }

    void test_soft_nms_sigma(bool is_caching_test) {
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
        topo.add(reorder("reformat_boxes", input_info("boxes"), this->layout_format, this->data_type),
                reorder("reformat_scores", input_info("scores"), this->layout_format, this->data_type),
                non_max_suppression("nms",
                                    input_info("reformat_boxes"),
                                    input_info("reformat_scores"),
                                    this->batch_size * this->classes_num * this->boxes_num,
                                    false,
                                    true,
                                    "num_per_class",
                                    "iou_threshold",
                                    "score_threshold",
                                    "soft_nms_sigma"));
        topo.add(reorder("plane_nms", input_info("nms"), format::bfyx, cldnn::data_types::i32));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        auto boxes_mem = this->get_boxes_memory(engine);
        auto scores_mem = this->get_scores_memory(engine);

        net->set_input_data("boxes", boxes_mem);
        net->set_input_data("scores", scores_mem);

        auto result = net->execute();

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
            ASSERT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
        }
    }
};

using nms_types = testing::Types<TypeWithLayoutFormat<float, cldnn::format::bfyx>,
                                 TypeWithLayoutFormat<float, cldnn::format::b_fs_yx_fsv32>,
                                 TypeWithLayoutFormat<float, cldnn::format::b_fs_yx_fsv16>,
                                 TypeWithLayoutFormat<float, cldnn::format::bs_fs_yx_bsv32_fsv16>,
                                 TypeWithLayoutFormat<float, cldnn::format::bs_fs_yx_bsv16_fsv16>,
                                 TypeWithLayoutFormat<float, cldnn::format::bs_fs_yx_bsv32_fsv32>,

                                 TypeWithLayoutFormat<ov::float16, cldnn::format::bfyx>,
                                 TypeWithLayoutFormat<ov::float16, cldnn::format::b_fs_yx_fsv32>,
                                 TypeWithLayoutFormat<ov::float16, cldnn::format::b_fs_yx_fsv16>,
                                 TypeWithLayoutFormat<ov::float16, cldnn::format::bs_fs_yx_bsv32_fsv16>,
                                 TypeWithLayoutFormat<ov::float16, cldnn::format::bs_fs_yx_bsv16_fsv16>,
                                 TypeWithLayoutFormat<ov::float16, cldnn::format::bs_fs_yx_bsv32_fsv32>>;

TYPED_TEST_SUITE(non_max_suppression_basic, nms_types);

TYPED_TEST(non_max_suppression_basic, basic) {
    this->test_basic(false);
}

TYPED_TEST(non_max_suppression_basic, num_per_class) {
    this->test_num_per_class(false);
}

TYPED_TEST(non_max_suppression_basic, optional_outputs) {
    this->test_optional_outputs(false);
}

TYPED_TEST(non_max_suppression_basic, multiple_outputs) {
    this->test_multiple_outputs(false);
}

TYPED_TEST(non_max_suppression_basic, iou_threshold) {
    this->test_iou_threshold(false);
}

TYPED_TEST(non_max_suppression_basic, score_threshold) {
    this->test_score_threshold(false);
}

TYPED_TEST(non_max_suppression_basic, nms_gather_score_threshold) {
    this->test_nms_gather_score_threshold(false);
}

TYPED_TEST(non_max_suppression_basic, soft_nms_sigma) {
    this->test_soft_nms_sigma(false);
}
#ifdef RUN_ALL_MODEL_CACHING_TESTS
TYPED_TEST(non_max_suppression_basic, basic_cached) {
    this->test_basic(true);
}

TYPED_TEST(non_max_suppression_basic, num_per_class_cached) {
    this->test_num_per_class(true);
}

TYPED_TEST(non_max_suppression_basic, optional_outputs_cached) {
    this->test_optional_outputs(true);
}

TYPED_TEST(non_max_suppression_basic, iou_threshold_cached) {
    this->test_iou_threshold(true);
}

TYPED_TEST(non_max_suppression_basic, score_threshold_cached) {
    this->test_score_threshold(true);
}

TYPED_TEST(non_max_suppression_basic, soft_nms_sigma_cached) {
    this->test_soft_nms_sigma(true);
}
#endif // RUN_ALL_MODEL_CACHING_TESTS
TYPED_TEST(non_max_suppression_basic, multiple_outputs_cached) {
    this->test_multiple_outputs(true);
}

namespace {
template<typename T, typename T_IND>
struct NmsRotatedParams {
    std::string test_name;
    int num_batches;
    int num_boxes;
    int num_classes;
    std::vector<T> boxes;
    std::vector<T> scores;
    int max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    bool sort_result_descending;
    bool clockwise;
    std::vector<T_IND> expected_indices;
    std::vector<T> expected_scores;
};

template <typename T> float getError();

template<>
float getError<float>() {
    return 0.001;
}

template<>
float getError<ov::float16>() {
    return 0.1;
}

template<typename T, typename T_IND>
struct nms_rotated_test : public ::testing::TestWithParam<NmsRotatedParams<T, T_IND>> {
public:
    void test(bool is_caching_test = false
    ) {
        const NmsRotatedParams<T, T_IND> param = testing::TestWithParam<NmsRotatedParams<T, T_IND>>::GetParam();
        const auto data_type = ov::element::from<T>();

        auto& engine = tests::get_test_engine();

        const auto boxes_layout = layout(ov::PartialShape{param.num_batches, param.num_boxes, 5}, data_type,
                                         format::bfyx);
        const auto scores_layout = layout(ov::PartialShape{param.num_batches, param.num_classes, param.num_boxes},
                                          data_type, format::bfyx);

        const int selected_indices_num = param.num_batches * param.num_classes * param.num_boxes;
        const auto selected_scores_layout = layout(ov::PartialShape{selected_indices_num/*expected_indices_count*/, 3},
                                                   data_type, format::bfyx);
        const auto valid_outputs_layout = layout(ov::PartialShape{1}, cldnn::data_types::i32, format::bfyx);

        const auto boxes_mem = engine.allocate_memory(boxes_layout);
        tests::set_values(boxes_mem, param.boxes);

        const auto scores_mem = engine.allocate_memory(scores_layout);
        tests::set_values(scores_mem, param.scores);

        const auto num_per_class_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
        tests::set_values(num_per_class_mem, {1.f * param.max_output_boxes_per_class});

        const auto iou_threshold_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
        tests::set_values(iou_threshold_mem, {param.iou_threshold});

        const auto score_threshold_mem = engine.allocate_memory(layout(data_types::f32, format::bfyx, tensor(batch(1))));
        tests::set_values(score_threshold_mem, {param.score_threshold});

        const auto selected_scores_mem = engine.allocate_memory(selected_scores_layout);
        const auto valid_outputs_mem = engine.allocate_memory(valid_outputs_layout);

        topology topo;
        topo.add(input_layout("boxes", boxes_layout));
        topo.add(input_layout("scores", scores_layout));
        topo.add(data("num_per_class", num_per_class_mem));
        topo.add(data("iou_threshold", iou_threshold_mem));
        topo.add(data("score_threshold", score_threshold_mem));
        topo.add(mutable_data("selected_scores", selected_scores_mem));
        topo.add(mutable_data("valid_outputs", valid_outputs_mem));
        auto nms = non_max_suppression("nms",
                                       input_info("boxes"),
                                       input_info("scores"),
                                       selected_indices_num,
                                       false,
                                       param.sort_result_descending,
                                       "num_per_class",
                                       "iou_threshold",
                                       "score_threshold",
                                       "",
                                       "selected_scores",
                                       "valid_outputs");
        nms.rotation = param.clockwise ? non_max_suppression::Rotation::CLOCKWISE :
                       non_max_suppression::Rotation::COUNTERCLOCKWISE;

        topo.add(nms);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);
        net->set_input_data("boxes", boxes_mem);
        net->set_input_data("scores", scores_mem);
        const auto result = net->execute();
        const auto indices_mem = result.at("nms").get_memory();
        const cldnn::mem_lock<T_IND> indices_ptr(indices_mem, get_test_stream());
        const cldnn::mem_lock<T> selected_scores_ptr(selected_scores_mem, get_test_stream());
        const cldnn::mem_lock<int> valid_outputs_ptr(valid_outputs_mem, get_test_stream());

        const auto expected_valid_outputs = param.expected_indices.size() / 3;
        const size_t num_valid_outputs = static_cast<size_t>(valid_outputs_ptr[0]);

        EXPECT_EQ(num_valid_outputs, expected_valid_outputs);
        ASSERT_GE(indices_ptr.size(), param.expected_indices.size());
        ASSERT_GE(selected_scores_ptr.size(), param.expected_scores.size());

        for (size_t i = 0; i < indices_ptr.size(); ++i) {
            if (i < num_valid_outputs * 3) {
                EXPECT_EQ(param.expected_indices[i], indices_ptr[i]) << "at i = " << i;
                EXPECT_NEAR(param.expected_scores[i], selected_scores_ptr[i], getError<T>()) << "at i = " << i;
            } else {
                EXPECT_EQ(indices_ptr[i], -1) << "at i = " << i;
                EXPECT_NEAR(selected_scores_ptr[i], -1, getError<T>()) << "at i = " << i;
            }
        }
    }
};


struct PrintToStringParamName {
    template<class T, class T_IND>
    std::string operator()(const testing::TestParamInfo<NmsRotatedParams<T, T_IND>>& info) {
        const auto& p = info.param;
        std::ostringstream result;
        result << p.test_name << "_";
        result << "DataType=" << ov::element::Type(ov::element::from<T>());
        result << "_IndexType=" << ov::element::Type(ov::element::from<T_IND>());
        return result.str();
    }
};


using nms_rotated_test_f32_i32 = nms_rotated_test<float, int32_t>;
using nms_rotated_test_f16_i32 = nms_rotated_test<ov::float16, int32_t>;

TEST_P(nms_rotated_test_f32_i32, basic) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(nms_rotated_test_f16_i32, basic) {
    ASSERT_NO_FATAL_FAILURE(test());
}

template<typename T, typename T_IND>
std::vector<NmsRotatedParams<T, T_IND>> getNmsRotatedParams() {
    const std::vector<NmsRotatedParams<T, T_IND>> params = {
            {"basic",
             1, 4, 1,
             std::vector<T>{
                7.0, 4.0, 8.0,  7.0,  0.5,
                4.0, 7.0, 9.0,  11.0, 0.6,
                4.0, 8.0, 10.0, 12.0, 0.3,
                2.0, 5.0, 13.0, 7.0,  0.6},
             std::vector<T>{0.65, 0.7, 0.55, 0.96},
             5000, 0.5f, 0.0f, false, true,
             std::vector<T_IND>{0, 0, 3, 0, 0, 1, 0, 0, 0},
             std::vector<T>{0.0, 0.0, 0.96, 0.0, 0.0, 0.7, 0.0, 0.0, 0.65},
            },
            {"max_out_2",
             1, 4, 1,
             std::vector<T>{
                7.0, 4.0, 8.0,  7.0,  0.5,
                4.0, 7.0, 9.0,  11.0, 0.6,
                4.0, 8.0, 10.0, 12.0, 0.3,
                2.0, 5.0, 13.0, 7.0,  0.6},
             std::vector<T>{0.65, 0.7, 0.55, 0.96},
             2, 0.5f, 0.0f, false, true,
             std::vector<T_IND>{0, 0, 3, 0, 0, 1},
             std::vector<T>{0.0, 0.0, 0.96, 0.0, 0.0, 0.7},
            },
            {"score_thresold",
             1, 4, 1,
             std::vector<T>{
                7.0, 4.0, 8.0,  7.0,  0.5,
                4.0, 7.0, 9.0,  11.0, 0.6,
                4.0, 8.0, 10.0, 12.0, 0.3,
                2.0, 5.0, 13.0, 7.0,  0.6},
             std::vector<T>{0.65, 0.7, 0.55, 0.96},
             5000, 0.5f, 0.67f, false, true,
             std::vector<T_IND>{0, 0, 3, 0, 0, 1},
             std::vector<T>{0.0, 0.0, 0.96, 0.0, 0.0, 0.7},
            },
            {"iou_thresold_2",
             1, 4, 1,
             std::vector<T>{
                7.0, 4.0, 8.0,  7.0,  0.5,
                4.0, 7.0, 9.0,  11.0, 0.6,
                4.0, 8.0, 10.0, 12.0, 0.3,
                2.0, 5.0, 13.0, 7.0,  0.6},
             std::vector<T>{0.65, 0.7, 0.55, 0.96},
             5000, 0.3f, 0.0f, false, true,
             std::vector<T_IND>{0, 0, 3, 0, 0, 0},
             std::vector<T>{0.0, 0.0, 0.96, 0.0, 0.0, 0.65},
            },
            {"negative_cw",
             1, 2, 1,
             std::vector<T>{6.0, 34.0, 4.0, 8.0, -0.7854, 9.0, 32, 2.0, 4.0, 0.0},
             std::vector<T>{0.8, 0.7},
             5000, 0.1f, 0.0f, false, true,
             std::vector<T_IND>{0, 0, 0, 0, 0, 1},
             std::vector<T>{0.0, 0.0, 0.8, 0.0, 0.0, 0.7}
            },
            {"negative_ccw",
             1, 2, 1,
             std::vector<T>{6.0, 34.0, 4.0, 8.0, -0.7854, 9.0, 32, 2.0, 4.0, 0.0},
             std::vector<T>{0.8, 0.7},
             5000, 0.1f, 0.0f, false, false,
             std::vector<T_IND>{0, 0, 0},
             std::vector<T>{0.0, 0.0, 0.8}
            },
            {"positive_ccw",
             1, 2, 1,
             std::vector<T>{6.0, 34.0, 4.0, 8.0, 0.7854, 9.0, 32, 2.0, 4.0, 0.0},
             std::vector<T>{0.8, 0.7},
             5000, 0.1f, 0.0f, false, false,
             std::vector<T_IND>{0, 0, 0, 0, 0, 1},
             std::vector<T>{0.0, 0.0, 0.8, 0.0, 0.0, 0.7}
            },
            {"positive_cw",
             1, 2, 1,
             std::vector<T>{6.0, 34.0, 4.0, 8.0, 0.7854, 9.0, 32, 2.0, 4.0, 0.0},
             std::vector<T>{0.8, 0.7},
             5000, 0.1f, 0.0f, false, true,
             std::vector<T_IND>{0, 0, 0},
             std::vector<T>{0.0, 0.0, 0.8}
            }
    };

    return params;
}
INSTANTIATE_TEST_SUITE_P(multiclass_nms_gpu_test,
                     nms_rotated_test_f32_i32,
                     ::testing::ValuesIn(getNmsRotatedParams<float, int32_t>()),
                     PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(multiclass_nms_gpu_test,
                     nms_rotated_test_f16_i32,
                     ::testing::ValuesIn(getNmsRotatedParams<ov::float16, int32_t>()),
                     PrintToStringParamName());
} // namespace
