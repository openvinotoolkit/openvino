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

#include <fstream>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

pass::DumpSorted::DumpSorted(const string& output_file)
    : m_output_file{output_file}
{
}

bool pass::DumpSorted::run_on_module(vector<shared_ptr<Function>>& functions)
{
    ofstream out{m_output_file};
    if (out)
    {
        for (shared_ptr<Function> f : functions)
        {
            out << "=====================================================================\n";
            out << f->get_name() << " start\n";
            out << "=====================================================================\n";
            for (const shared_ptr<Node>& node : f->get_ordered_ops())
            {
                out << node->get_name() << "(";
                vector<string> inputs;
                for (auto& input : node->inputs())
                {
                    inputs.push_back(input.get_tensor().get_name());
                }
                out << join(inputs);
                out << ") -> ";

                vector<string> outputs;
                for (auto& output : node->outputs())
                {
                    outputs.push_back(output.get_tensor().get_name());
                }
                out << join(outputs);
                out << "\n";

                for (const descriptor::Tensor* tensor : node->liveness_new_list)
                {
                    out << "    N " << tensor->get_name() << "\n";
                }
                for (const descriptor::Tensor* tensor : node->liveness_free_list)
                {
                    out << "    F " << tensor->get_name() << "\n";
                }
            }
            out << "=====================================================================\n";
            out << f->get_name() << " end\n";
            out << "=====================================================================\n";
        }
    }

    return false;
}
