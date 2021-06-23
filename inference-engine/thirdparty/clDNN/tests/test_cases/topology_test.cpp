// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <cldnn/primitives/lrn.hpp>
#include <cldnn/primitives/convolution.hpp>
#include <cldnn/primitives/fully_connected.hpp>
#include <cldnn/primitives/pooling.hpp>
#include <cldnn/primitives/data.hpp>
#include <cldnn/primitives/reorder.hpp>
#include <cldnn/primitives/scale.hpp>
#include <cldnn/primitives/eltwise.hpp>
#include <cldnn/primitives/softmax.hpp>
#include <cldnn/primitives/activation.hpp>
#include <cldnn/primitives/concatenation.hpp>

#include <include/topology_impl.h>

#include <iostream>
#include <deque>
#include <set>

typedef std::tuple<cldnn::layout, std::vector<unsigned>> topology_params;

void PrintTupleTo(const topology_params& t, ::std::ostream* os)
{
    const auto & output_layout = std::get<0>(t);
    const auto & generator = std::get<1>(t);
    std::stringstream ss;

    ss << "Topology test failed: ("
        << cldnn::data_type_traits::name(output_layout.data_type) << " "
        << tests::test_params::print_tensor(output_layout.size) << ") Generator: [";
    for (auto v : generator)
    {
        ss << v << ", ";
    }
    ss.seekp(-1, ss.cur) << "]\n";
    *os << ss.str();
}

class topology_test : public ::testing::TestWithParam<topology_params>
{
protected:
    class topology_generator
    {
    public:
        typedef std::pair<cldnn::primitive_id, cldnn::layout> named_layout;
        class topology_layer_type
        {
        public:
            // return false for invalid output_layout
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts) = 0;
            virtual ~topology_layer_type() = default;
        };
        static std::vector<std::shared_ptr<topology_layer_type>> layer_types;
        static cldnn::topology* CreateTopology(cldnn::layout output_layout, const std::vector<unsigned> generator_vec)
        {
            if (generator_vec.size() < 2)
            {
                return nullptr;
            }
            auto topology = new cldnn::topology();
            std::deque<named_layout> inputs;
            const unsigned max_index = generator_vec[0];
            inputs.push_back({ output_layer_id, output_layout });
            for (unsigned g_index = 1; g_index < generator_vec.size(); g_index++)
            {
                auto input = inputs.front();
                inputs.pop_front();
                if (!AddSinglePrimitive(*topology, input.first, input.second, inputs, generator_vec.at(g_index), max_index))
                {
                    delete topology;
                    return nullptr;
                }
            }
            // add data inputs
            for (const auto& input : inputs)
            {
                //first add a reorder to enable optimize_data
                cldnn::primitive_id input_data_id = input.first + "_input";
                topology->add(cldnn::reorder(input.first, input_data_id, input.second));
                AddRandomMemory(*topology, input_data_id, input.second);
            }
            return topology;
        }
        static cldnn::primitive_id CreateLayerId()
        {
            static unsigned layer_id = 0;
            return "tg_layer_" + std::to_string(layer_id++);
        }
        static const cldnn::primitive_id output_layer_id;
        static bool AddSinglePrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts, unsigned type_index, unsigned max_type)
        {
            if (layer_types.size() < max_type)
            {
                return false;//shouldn't happen
            }
            for (unsigned t = 0; t < max_type; t++)
            {
                if (layer_types.at((type_index + t) % max_type)->AddPrimitive(topology, id, output_layout, input_layouts))
                {
                    return true;
                }
            }
            //todo: consider using a data primitive here
            return false;
        }
        static void AddRandomMemory(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout layout)
        {
            //todo: allocate mem, randomize values by type, add to topology
            auto mem_primitive = topology_test::engine.allocate_memory(layout);
            switch (layout.data_type)
            {
            case cldnn::data_types::f32:
                tests::set_random_values<float>(mem_primitive, true, 10, 1);
                break;
            case cldnn::data_types::f16:
                tests::set_random_values<FLOAT16>(mem_primitive, true, 4, 1);
                break;
            default:
                assert(0);
            }
            topology.add(cldnn::data(id, mem_primitive));
        }
    protected:
        topology_generator() {}

        class convolution_layer_type : public topology_layer_type
        {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts)
            {
                if (output_layout.format != cldnn::format::bfyx)
                {
                    return false;
                }
                // for now using just one set of params
                // todo: randomize params
                cldnn::primitive_id weights_id = id + "_weights";
                cldnn::layout weights_layout(output_layout.data_type,
                cldnn::format::yxfb,{ output_layout.size.feature[0], output_layout.size.feature[0], 1, 1 });
                AddRandomMemory(topology, weights_id, weights_layout);
                cldnn::primitive_id bias_id = id + "_bias";
                cldnn::layout bias_layout(output_layout.data_type,
                cldnn::format::bfyx,{ 1, 1, output_layout.size.feature[0], 1 });
                AddRandomMemory(topology, bias_id, bias_layout);

                cldnn::primitive_id input_id = topology_generator::CreateLayerId();
                input_layouts.push_back({ input_id, output_layout });
                topology.add(
                    cldnn::convolution(id, input_id, { weights_id }, { bias_id }));
                return true;
            }
        };
        class normalization_layer_type : public topology_layer_type
        {
            bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts)
            {
                if (output_layout.format != cldnn::format::bfyx)
                {
                    return false;
                }
                // for now using just one set of params
                // todo: randomize params
                cldnn::primitive_id input_id = topology_generator::CreateLayerId();
                input_layouts.push_back({ input_id, output_layout });
                uint32_t size = 5;
                float k = 1.0f;
                float alpha = 0.0001f;
                float beta = 0.75f;
                cldnn::lrn_norm_region norm_type = cldnn::lrn_norm_region_across_channel;
                topology.add(cldnn::lrn(id, input_id, size, k, alpha, beta, norm_type));
                return true;
            }
        };
        class pooling_layer_type : public topology_layer_type
        {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts)
            {
                if (output_layout.size.spatial.size() != 2)
                {
                    return false;
                }
                // for now using just one set of params
                // todo: randomize params
                cldnn::primitive_id input_id = topology_generator::CreateLayerId();
                cldnn::pooling_mode mode = cldnn::pooling_mode::max;
                cldnn::tensor stride = { 1, 1, 1, 1 };
                cldnn::tensor size = { 1, 1, 3, 3 };
                input_layouts.push_back({ input_id, output_layout });
                topology.add(cldnn::pooling(id, input_id, mode, stride, size));
                return true;
            }
        };
        class fully_connected_layer_type : public topology_layer_type
        {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts)
            {
                if (output_layout.format != cldnn::format::bfyx)
                {
                    return false;
                }

                // for now using just one set of params
                // todo: randomize params

                cldnn::layout input_layout(output_layout.data_type, cldnn::format::bfyx,{ output_layout.size.batch[0] , output_layout.size.feature[0], 100, 100 } );
                cldnn::primitive_id weights_id = id + "_weights";
                cldnn::layout weights_layout(output_layout.data_type,
                cldnn::format::bfyx,{ output_layout.size.feature[0], input_layout.size.feature[0], input_layout.size.spatial[0], input_layout.size.spatial[1] });
                AddRandomMemory(topology, weights_id, weights_layout);
                cldnn::primitive_id bias_id = id + "_bias";
                cldnn::layout bias_layout(output_layout.data_type,
                cldnn::format::bfyx,{ 1, 1, output_layout.size.feature[0], 1 });
                AddRandomMemory(topology, bias_id, bias_layout);

                cldnn::primitive_id input_id = topology_generator::CreateLayerId();
                input_layouts.push_back({ input_id, input_layout });
                topology.add(
                    cldnn::fully_connected(id, input_id, { weights_id }, { bias_id }));
                return true;
            }
        };
        class reorder_layer_type : public topology_layer_type
        {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts)
            {
                // for now using just one set of params
                // todo: randomize params
                cldnn::primitive_id input_id = topology_generator::CreateLayerId();
                input_layouts.push_back({ input_id, output_layout });//empty reorder
                topology.add(cldnn::reorder(id,input_id,output_layout));
                return true;
            }
        };
        class activation_layer_type : public topology_layer_type
        {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts)
            {
                // for now using just one set of params
                // todo: randomize params
                cldnn::primitive_id input_id = topology_generator::CreateLayerId();
                input_layouts.push_back({ input_id, output_layout });
                topology.add(cldnn::activation(id, input_id, cldnn::activation_func::relu));
                return true;
            }
        };
        class depth_concatenate_layer_type : public topology_layer_type
        {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts)
            {
                // for now using just one set of params
                // todo: randomize params
                if (output_layout.format != cldnn::format::bfyx// should be "output_layout.size.format.dimension() < 4" but requires too many case handling since tensor is immutable
                    || output_layout.size.feature[0] < 2)
                {
                    return false;
                }
                cldnn::primitive_id input_id1 = topology_generator::CreateLayerId();
                cldnn::primitive_id input_id2 = topology_generator::CreateLayerId();
                cldnn::layout input_layout1(
                    output_layout.data_type,
                    cldnn::format::bfyx,
                        {
                            output_layout.size.batch[0],
                            output_layout.size.feature[0] - 1,
                            output_layout.size.spatial[0],
                            output_layout.size.spatial[1]
                        }
                );
                cldnn::layout input_layout2(
                    output_layout.data_type,
                    cldnn::format::bfyx,
                        {
                            output_layout.size.batch[0],
                            1,
                            output_layout.size.spatial[0],
                            output_layout.size.spatial[1]
                        }
                );
                input_layouts.push_back({ input_id1, input_layout1 });
                input_layouts.push_back({ input_id2, input_layout2 });

                topology.add(cldnn::concatenation(id, { input_id1,input_id2 }, cldnn::concatenation::along_f));
                return true;
            }
        };
        class eltwise_layer_type : public topology_layer_type
        {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts)
            {
                // for now using just one set of params
                // todo: randomize params
                cldnn::primitive_id input_id = topology_generator::CreateLayerId();
                input_layouts.push_back({ input_id, output_layout });
                cldnn::primitive_id eltwise_params_id = id + "_eltwise_params";
                AddRandomMemory(topology, eltwise_params_id, output_layout);
                topology.add(cldnn::eltwise(id, {input_id, eltwise_params_id}, cldnn::eltwise_mode::max));
                return true;
            }
        };
        class scale_layer_type : public topology_layer_type
        {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts)
            {
                // for now using just one set of params
                // todo: randomize params
                cldnn::primitive_id input_id = topology_generator::CreateLayerId();
                input_layouts.push_back({ input_id, output_layout });
                cldnn::primitive_id scale_params_id = id + "_scale_params";
                AddRandomMemory(topology, scale_params_id, output_layout);
                topology.add(cldnn::scale(id, input_id, scale_params_id, ""));
                return true;
            }
        };
        class softmax_layer_type : public topology_layer_type
        {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts)
            {
                // for now using just one set of params
                // todo: randomize params
                cldnn::primitive_id input_id = topology_generator::CreateLayerId();
                input_layouts.push_back({ input_id, output_layout });
                topology.add(cldnn::softmax(id, input_id));
                return true;
            }
        };
/* missing layers
        class batch_norm_layer_type : public topology_layer_type {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts) {
            }
        };
        class crop_layer_type : public topology_layer_type {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts) {
            }
        };
        class deconvolution_layer_type : public topology_layer_type {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts) {
            }
        };
        class prior_box_layer_type : public topology_layer_type {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts) {
            }
        };
        class roi_pooling_layer_type : public topology_layer_type {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts) {
            }
        };
        class psroi_pooling_layer_type : public topology_layer_type {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts) {
            }
        };
        class proposal_layer_type : public topology_layer_type {
            virtual bool AddPrimitive(cldnn::topology& topology, cldnn::primitive_id id, cldnn::layout output_layout, std::deque<named_layout>& input_layouts) {
            }
        };
*/
    };
public:
    static const unsigned topologies_per_type_size = 10;
    topology_test() : output_layout(std::get<0>(GetParam())), generator(std::get<1>(GetParam())) {}
    void run_single_test()
    {
        cldnn::topology* topology = topology_generator::CreateTopology(output_layout, generator);
        EXPECT_NE(topology, nullptr);
        cldnn::build_options options;
        options.set_option(cldnn::build_option::optimize_data(true));
        auto& engine = tests::get_test_engine();
        cldnn::network network(engine, *topology, options);
        auto outputs = network.execute();
        EXPECT_NE(outputs.find(topology_generator::output_layer_id), outputs.end());

        delete topology;
    }

    static std::vector<cldnn::layout> generate_all_output_layouts()
    {
        assert(all_output_layouts.empty());
        std::vector<cldnn::data_types> data_types = { cldnn::data_types::f32, cldnn::data_types::f16 };
        std::vector<cldnn::tensor> output_tensors = {
            { 1, 1, 100, 1 },
            { 5, 1, 100, 1 },
            { 1, 10, 100, 100 },
            { 8, 1, 100, 100 },
        };
        // todo: consider iterating on format X dimensions

        for (auto dt : data_types) {
            for (auto t : output_tensors) {
                all_output_layouts.push_back(cldnn::layout(dt, cldnn::format::bfyx, t));
            }
        }
        return all_output_layouts;
    }
    template<unsigned generator_length>
    static std::set<std::vector<unsigned>> all_generator_vectors()
    {
        // create vectors used to create topologies [max_layer_index, layer_index0, layer_index1,...]
        std::set<std::vector<unsigned>> all_generators;
        static std::default_random_engine rng(tests::random_seed);
        std::uniform_int_distribution<unsigned> distribution(0, 0xFF);//assuming we won't exceed 256 total layer types

        const unsigned Initial_layer_types = 10;//don't change this - starting with this index ensures adding layers won't alter previously generated tests
        for (unsigned types = Initial_layer_types; types <= topology_test::topology_generator::layer_types.size(); types++)
        {
            for (unsigned i = 0; i < topologies_per_type_size; i++)
            {
                std::vector<unsigned> generator;
                generator.push_back(types);
                for (unsigned j = 0; j < generator_length; j++)
                {
                    generator.push_back(distribution(rng) % types);
                }
                all_generators.insert(generator);
            }
        }
        return all_generators;
    }
    static void TearDownTestCase() { }
    static std::string custom_param_name(const ::testing::TestParamInfo<topology_params>& info)
    {
        const auto & output_layout = std::get<0>(info.param);
        const auto & generator = std::get<1>(info.param);
        std::stringstream ss;
        ss << info.index << "_";
        for (auto v : generator)
        {
            ss << v << "_";
        }
        ss << cldnn::data_type_traits::name(output_layout.data_type) << "_";
        ss << cldnn::format::traits(output_layout.format).order;
        for (const auto& d : output_layout.size.raw)
        {
            ss << "_" << d;
        }

        return ss.str();
    }
protected:
    cldnn::layout output_layout;
    std::vector<unsigned> generator;

    static cldnn::engine& engine;
    static std::vector<cldnn::layout> all_output_layouts;//just for tear-down
};

cldnn::engine& topology_test::engine = tests::get_test_engine();
std::vector<cldnn::layout> topology_test::all_output_layouts = {};

std::vector<std::shared_ptr<topology_test::topology_generator::topology_layer_type>> topology_test::topology_generator::layer_types = {
    std::shared_ptr<topology_test::topology_generator::topology_layer_type>(new topology_test::topology_generator::normalization_layer_type()),
    std::shared_ptr<topology_test::topology_generator::topology_layer_type>(new topology_test::topology_generator::pooling_layer_type()),
    std::shared_ptr<topology_test::topology_generator::topology_layer_type>(new topology_test::topology_generator::convolution_layer_type()),
    std::shared_ptr<topology_test::topology_generator::topology_layer_type>(new topology_test::topology_generator::fully_connected_layer_type()),
    std::shared_ptr<topology_test::topology_generator::topology_layer_type>(new topology_test::topology_generator::reorder_layer_type()),
    std::shared_ptr<topology_test::topology_generator::topology_layer_type>(new topology_test::topology_generator::activation_layer_type()),
    std::shared_ptr<topology_test::topology_generator::topology_layer_type>(new topology_test::topology_generator::depth_concatenate_layer_type()),
    std::shared_ptr<topology_test::topology_generator::topology_layer_type>(new topology_test::topology_generator::eltwise_layer_type()),
    std::shared_ptr<topology_test::topology_generator::topology_layer_type>(new topology_test::topology_generator::scale_layer_type()),
    std::shared_ptr<topology_test::topology_generator::topology_layer_type>(new topology_test::topology_generator::softmax_layer_type()),
    // Only add new types at the end
};
const cldnn::primitive_id topology_test::topology_generator::output_layer_id("tg_output_layer");

TEST_P(topology_test, TOPOLOGY)
{
     try
     {
         run_single_test();
         if (::testing::Test::HasFailure())
         {
             PrintTupleTo(GetParam(), &std::cout);
         }
     }
     catch (...)
     {
         PrintTupleTo(GetParam(), &std::cout);
         throw;
     }
}

INSTANTIATE_TEST_SUITE_P(DISABLED_TOPOLOGY,
    topology_test,
    ::testing::Combine( ::testing::ValuesIn(topology_test::generate_all_output_layouts()),
                        ::testing::ValuesIn(topology_test::all_generator_vectors<3>())),
    topology_test::custom_param_name);
