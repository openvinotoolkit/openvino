//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"

// #include "ngraph/codegen/compiler.hpp"
// #include "ngraph/file_util.hpp"
// #include "ngraph/log.hpp"
// #include "ngraph/util.hpp"

// using namespace std;
// using namespace ngraph;

// TEST(DISABLED_include, complete)
// {
//     vector<string> include_files;
//     set<string> ext_list{".hpp"};
//     set<string> exclude{"onnx_import", "onnxifi", "intelgpu", "op_tbl.hpp"};
//     auto func = [&](const std::string& file, bool is_dir) {
//         if (!is_dir && file.size() > 4)
//         {
//             for (const string& x : exclude)
//             {
//                 if (file.find(x) != file.npos)
//                 {
//                     return;
//                 }
//             }
//             string ext = file.substr(file.size() - 4);
//             if (ext_list.find(ext) != ext_list.end())
//             {
//                 include_files.push_back(file);
//             }
//         }
//     };
//     file_util::iterate_files(NGRAPH_INCLUDES, func, true);

//     for (const string& include : include_files)
//     {
//         string source = "#include <" + include + ">\n ";

//         codegen::Compiler compiler;
//         compiler.add_header_search_path(JSON_INCLUDES);
//         auto module = compiler.compile(source);
//         if (!module)
//         {
//             cout << "fail " << include << endl;
//         }
//     }
// }
