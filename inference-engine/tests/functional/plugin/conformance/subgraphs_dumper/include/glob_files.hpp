//// Copyright (C) 2019-2020 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//#pragma once
//
//#include <utility>
//#include <vector>
//#include <string>
//#include <regex>
//#include <filesystem>
//#include <any>
//#include <inference_engine.hpp>
//
//namespace SubgraphsDumper {
//
//class FolderIterator {
//public:
//    explicit FolderIterator(const std::string& root_folder, const std::string& search_pattern = ".*");
//    std::vector<std::string> get_folder_content() {return m_folder_content;}
//    std::vector<std::string> get_folder_content(const std::string& search_pattern);
//private:
//    std::vector<std::string> m_folder_content;
//};
//}  // namespace SubgraphsDumper