// TODO: c++17 code
//// Copyright (C) 2019-2020 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//#include <glob_files.hpp>
//#include <sstream>
//#include <iostream>
//
//using namespace SubgraphsDumper;
//namespace fs = std::filesystem;
//FolderIterator::FolderIterator(const std::string &root_folder, const std::string &search_pattern) {
//    std::regex re(search_pattern);
//    for (const fs::directory_entry &p : fs::recursive_directory_iterator(root_folder)) {
//        if (fs::is_directory(p.path())) {
//            continue;
//        }
//        if (!std::regex_match(p.path().string(), re)) {
//            continue;
//        }
//        m_folder_content.push_back(p.path().string());
//    }
//}
//
//std::vector<std::string> FolderIterator::get_folder_content(const std::string &search_pattern) {
//    std::vector<std::string> filtered_files;
//    std::regex re(search_pattern);
//    for (const auto &f : m_folder_content) {
//        if (std::regex_match(f, re)) {
//            filtered_files.push_back(f);
//        }
//    }
//    return filtered_files;
//}