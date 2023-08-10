// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <string>
#include <vector>

class MyProfile {
public:
    MyProfile() = delete;
    // MyProfile(MyProfile&)=delete;
    MyProfile(const std::string& name,
              const std::vector<std::pair<std::string, std::string>>& args =
                  std::vector<std::pair<std::string, std::string>>());
    ~MyProfile();

private:
    std::string _name;
    uint64_t _ts1;
    std::vector<std::pair<std::string, std::string>> _args;
};

#define MY_PROFILE(NAME)           MyProfile(NAME + std::string(":") + std::to_string(__LINE__))
#define MY_PROFILE_ARGS(NAME, ...) MyProfile(NAME + std::string(":") + std::to_string(__LINE__), __VA_ARGS__)

// Example:
// ==========================================
// auto p = MyProfile("fun_name")
// Or
// {
//     auto p = MyProfile("fun_name")
//     func()
// }
// Or
// {
//     auto p2 = MyProfile("fun_name", {{"arg1", "sleep 30 ms"}});
//     func()
// }