// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/core.hpp"
#include "common_utils.h"


class InferApiBase {
public:
    virtual void load_plugin(const std::string &device) = 0;

    virtual void unload_plugin(const std::string &device) = 0;

    virtual void read_network(const std::string &model) = 0;

    virtual void load_network(const std::string &device) = 0;

    virtual void create_infer_request() = 0;

    virtual void create_and_infer(const bool &aysnc) = 0;

    virtual void infer() = 0;

    virtual void prepare_input() = 0;

    virtual void change_batch_size(int multiplier, int cur_iter) = 0;

    virtual void set_input_params(const std::string &model) = 0;

    virtual void set_config(const std::string &device, const ov::AnyMap &properties) = 0;

    virtual unsigned int get_property(const std::string &name) = 0;
};

class InferAPI2 : public InferApiBase {
public:
    InferAPI2();

    void load_plugin(const std::string &device) override;

    void unload_plugin(const std::string &device) override;

    void read_network(const std::string &model) override;

    void load_network(const std::string &device) override;

    void create_infer_request() override;

    void create_and_infer(const bool &aysnc) override;

    void prepare_input() override;

    void infer() override;

    void change_batch_size(int multiplier, int cur_iter) override;

    void set_input_params(const std::string &model) override;

    void set_config(const std::string &device, const ov::AnyMap &properties) override;

    unsigned int get_property(const std::string &name) override;

private:
    ov::Core ie;
    std::shared_ptr<ov::Model> network;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    std::vector<ov::Output<ov::Node>> inputs;
    std::vector<ov::Output<ov::Node>> outputs;
    int original_batch_size;
    std::map<std::string, ov::Any> config;
};

std::shared_ptr<InferApiBase> create_infer_api_wrapper();
