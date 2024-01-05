// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <string>
#include <iostream>
#include <future>
#include "openvino/openvino.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/reshape.hpp"
#include "thread_local.hpp"

std::shared_ptr<ov::Model> makeSplitConcat(std::vector<size_t> inputShape = {1, 4, 24, 24},
                                           ov::element::Type_t type = ov::element::Type_t::f32);

void core_get_property_test(std::string target_device)
{
    std::promise<void> call_finish_promise;
    std::future<void> call_finish_future = call_finish_promise.get_future();
    std::promise<void> thread_exit_promise;
    std::future<void> thread_exit_future = thread_exit_promise.get_future();
    std::thread sub_thread;
    {
        ov::Core ie;
        sub_thread = std::thread([&]
                                 {
            ie.get_property(target_device, ov::supported_properties);
            call_finish_promise.set_value();
            thread_exit_future.get(); });
        call_finish_future.get();
    }
    thread_exit_promise.set_value();
    if (sub_thread.joinable())
    {
        sub_thread.join();
    }
}

std::shared_ptr<ov::Model> makeSplitConcat(std::vector<size_t> inputShape, ov::element::Type_t type)
{
    auto param1 = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{inputShape});
    param1->set_friendly_name("Param1");
    param1->output(0).get_tensor().set_names({"data1"});
    auto axis_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto split = std::make_shared<ov::op::v1::Split>(param1, axis_node, 2);
    split->set_friendly_name("Split");
    split->output(0).get_tensor().set_names({"tensor_split_1"});
    split->output(1).get_tensor().set_names({"tensor_split_2"});

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{split->output(0), split->output(1)}, 1);
    concat->set_friendly_name("Concat_op");
    concat->output(0).get_tensor().set_names({"Concat"});
    auto result = std::make_shared<ov::op::v0::Result>(concat);
    result->set_friendly_name("Result");
    auto fn_ptr = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1});
    fn_ptr->set_friendly_name("SplitConcat");
    return fn_ptr;
}

void core_infer_test(std::string target_device)
{
    std::promise<void> call_finish_promise;
    std::future<void> call_finish_future = call_finish_promise.get_future();
    std::promise<void> thread_exit_promise;
    std::future<void> thread_exit_future = thread_exit_promise.get_future();
    std::thread sub_thread;
    auto actualNetwork = makeSplitConcat();
    {
        ov::Core ie;
        auto net = ie.compile_model(actualNetwork, target_device);
        auto infer_req = net.create_infer_request();
        sub_thread = std::thread([&]
                                 {
            ie.get_property(target_device, ov::supported_properties);
            infer_req.infer();
            call_finish_promise.set_value();
            thread_exit_future.get(); });
        call_finish_future.get();
    }
    thread_exit_promise.set_value();
    if (sub_thread.joinable())
    {
        sub_thread.join();
    }
}
