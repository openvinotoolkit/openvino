/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "instrumentation.h"

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <iostream>


namespace instrumentation {
    // initalize dumping directory for whole run
    const std::string logger::dump_dir = DUMP_DIRECTORY;

    static float convert_half_to_float(half_t val, bool flush_denorm_to_zero = false)
    {
#if defined HALF_HALF_HPP
        return val;
#else
        // FP32 parts extracted from FP16.
        uint32_t sign = (static_cast<uint16_t>(val) & 0x8000U) << 16;
        uint32_t mantissa = (static_cast<uint16_t>(val) & 0x3FFU) << 13;

        uint32_t exp_val_f16 = (static_cast<uint16_t>(val) & 0x7C00U) >> 10;
        uint32_t exp;
        if (exp_val_f16 == 0)
        {
            // Handling +/-0 and denormals.
            if (mantissa == 0)
            {
                exp = 0;
            }
            else if (flush_denorm_to_zero)
            {
                sign = 0;
                exp = 0;
                mantissa = 0;
            }
            else
            {
                // Denorms conversion to normal numbers.
                exp = 127 - 15;
                while (!(mantissa & 0x400000U))
                {
                    mantissa <<= 1;
                    --exp;
                }
                mantissa = (mantissa << 1) & 0x7FFFFFU;
                exp <<= 23;
            }
        }
        else
        {
            // Handling +/-infinity, NaN and normal numbers.
            exp = (exp_val_f16 == 0x1FU ? 0xFFU : exp_val_f16 + 127 - 15) << 23;
        }

        float ret;
        reinterpret_cast<uint32_t&>(ret) = sign | exp | mantissa;

        return ret;
#endif
    }

    float convert_element(float f)
    {
        return f;
    }

    float convert_element(half_t h)
    {
        return convert_half_to_float(h);
    }

    template<typename elemType>
    void dump_byxf(const cldnn::memory& mem, bool single_batch, cldnn::tensor::value_type batch_id, bool single_feature, cldnn::tensor::value_type feature_id, std::vector<std::vector<std::stringstream>> & streams)
    {
        auto mem_arg = mem.get_layout();
        auto mem_ptr = mem.pointer<elemType>();

        unsigned int input_it = 0;
        for (cldnn::tensor::value_type b = 0; b < mem_arg.size.batch[0]; b++)
        {
            for (cldnn::tensor::value_type y = 0; y < mem_arg.size.spatial[1]; y++)
            {
                for (cldnn::tensor::value_type x = 0; x < mem_arg.size.spatial[0]; x++)
                {
                    for (cldnn::tensor::value_type f = 0; f < mem_arg.size.feature[0]; f++)
                    {
                        if ((!single_batch || b == batch_id) && (!single_feature || f == feature_id))
                        {
                            streams[b][f] << convert_element(mem_ptr[input_it]) << " ";
                            if (x == mem_arg.size.spatial[0] - 1)
                            {
                                streams[b][f] << std::endl;
                            }
                        }
                        input_it++;
                    }
                }
            }
        }
    }

    template<typename elemType>
    void dump_bfyx(const cldnn::memory& mem, bool single_batch, cldnn::tensor::value_type batch_id, bool single_feature, cldnn::tensor::value_type feature_id, std::vector<std::vector<std::stringstream>> & streams)
    {
        auto mem_arg = mem.get_layout();
        auto mem_ptr = mem.pointer<elemType>();

        unsigned int input_it = 0;
        for (cldnn::tensor::value_type b = 0; b < mem_arg.size.batch[0]; b++)
        {
            for (cldnn::tensor::value_type f = 0; f < mem_arg.size.feature[0]; f++)
            {
                for (cldnn::tensor::value_type y = 0; y < mem_arg.size.spatial[1]; y++)
                {
                    for (cldnn::tensor::value_type x = 0; x < mem_arg.size.spatial[0]; x++)
                    {
                        if ((!single_batch || b == batch_id) && (!single_feature || f == feature_id))
                        {
                            streams[b][f] << convert_element(mem_ptr[input_it]) << " ";
                            if (x == mem_arg.size.spatial[0] - 1)
                            {
                                streams[b][f] << std::endl;
                            }
                        }
                        input_it++;
                    }
                }
            }
        }
    }

    template<typename elemType>
    void dump_yxfb(const cldnn::memory& mem, bool single_batch, cldnn::tensor::value_type batch_id, bool single_feature, cldnn::tensor::value_type feature_id, std::vector<std::vector<std::stringstream>> & streams)
    {
        auto mem_arg = mem.get_layout();
        auto mem_ptr = mem.pointer<elemType>();

        unsigned int input_it = 0;
        for (cldnn::tensor::value_type y = 0; y < mem_arg.size.spatial[1]; y++)
        {
            for (cldnn::tensor::value_type x = 0; x < mem_arg.size.spatial[0]; x++)
            {
                for (cldnn::tensor::value_type f = 0; f < mem_arg.size.feature[0]; f++)
                {
                    for (cldnn::tensor::value_type b = 0; b < mem_arg.size.batch[0]; b++)
                    {
                        if ((!single_batch || b == batch_id) && (!single_feature || f == feature_id))
                        {
                            streams[b][f] << convert_element(mem_ptr[input_it]) << " ";
                            if (x == mem_arg.size.spatial[0] - 1)
                            {
                                streams[b][f] << std::endl;
                            }
                        }
                        input_it++;
                    }
                }
            }
        }
    }

    template<typename elemType>
    void dump_xb(const cldnn::memory& mem, bool single_batch, cldnn::tensor::value_type batch_id, std::vector<std::vector<std::stringstream>> & streams)
    {
        auto mem_arg = mem.get_layout();
        auto mem_ptr = mem.pointer<elemType>();

        unsigned int input_it = 0;
        for (cldnn::tensor::value_type x = 0; x < mem_arg.size.spatial[0]; x++)
        {
            for (cldnn::tensor::value_type b = 0; b < mem_arg.size.batch[0]; b++)
            {
                if (!single_batch || b == batch_id)
                {
                    streams[b][0] << convert_element(mem_ptr[input_it]) << std::endl;
                }
                input_it++;
            }
        }
    }

    template<typename elemType>
    void dump_bx(const cldnn::memory& mem, bool single_batch, cldnn::tensor::value_type batch_id, std::vector<std::vector<std::stringstream>> & streams)
    {
        auto mem_arg = mem.get_layout();
        auto mem_ptr = mem.pointer<elemType>();

        unsigned int input_it = 0;
        for (cldnn::tensor::value_type b = 0; b < mem_arg.size.batch[0]; b++)
        {
            for (cldnn::tensor::value_type x = 0; x < mem_arg.size.spatial[0]; x++)
            {
                if (!single_batch || b == batch_id)
                {
                    streams[b][0] << convert_element(mem_ptr[input_it]) << std::endl;
                }
                input_it++;
            }
        }
    }

    template<typename elemType>
    void dump_yxio(const cldnn::memory& mem, std::stringstream & stream)
    {
        auto mem_arg = mem.get_layout();
        auto mem_ptr = mem.pointer<elemType>();

        auto i_size = mem_arg.size.batch[0];
        auto o_size = mem_arg.size.feature[0];
        auto x_size = mem_arg.size.spatial[0];
        auto y_size = mem_arg.size.spatial[1];
        unsigned int input_it = 0;
        for (cldnn::tensor::value_type o = 0; o < o_size; o++)
        {
            for (cldnn::tensor::value_type i = 0; i < i_size; i++)
            {
                for (cldnn::tensor::value_type x = 0; x < x_size; x++)
                {
                    for (cldnn::tensor::value_type y = 0; y < y_size; y++)
                    {
                        stream<< convert_element(mem_ptr[input_it]) << " ";
                        input_it++;
                    }
                    stream<< std::endl;
                }
            }
        }
    }

    template<typename elemType>
    void dump_oiyx(const cldnn::memory& mem, std::stringstream & stream)
    {
        auto mem_arg = mem.get_layout();
        auto mem_ptr = mem.pointer<elemType>();

        auto i_size = mem_arg.size.batch[0];
        auto o_size = mem_arg.size.feature[0];
        auto x_size = mem_arg.size.spatial[0];
        auto y_size = mem_arg.size.spatial[1];
        unsigned int input_it = 0;
        for (cldnn::tensor::value_type x = 0; x < x_size; x++)
        {
            for (cldnn::tensor::value_type y = 0; y < y_size; y++)
            {
                for (cldnn::tensor::value_type i = 0; i < i_size; i++)
                {
                    for (cldnn::tensor::value_type o = 0; o < o_size; o++)
                    {
                        stream << convert_element(mem_ptr[input_it]) << " ";
                        input_it++;
                    }
                    stream << std::endl;
                }
            }
        }
    }

    template<typename elemType>
    void dump_os_iyx_osv16(const cldnn::memory& mem, std::stringstream & stream)
    {
        auto mem_arg = mem.get_layout();
        auto mem_ptr = mem.pointer<elemType>();

        auto i_size = mem_arg.size.batch[0];
        auto o_size = mem_arg.size.feature[0];
        auto x_size = mem_arg.size.spatial[0];
        auto y_size = mem_arg.size.spatial[1];
        auto weights_size = i_size * o_size * x_size * y_size; //count() also counts feature[1]
        int slice_value = 16;
        cldnn::tensor::value_type it = 0;
        while (it < weights_size)
        {
            stream << convert_element(mem_ptr[it]) << " ";
            it++;
            if (it % slice_value == 0) //separate every bsv with a new line
                stream << std::endl;
        };
    }

    template<typename elemType>
    void dump_bs_xs_xsv8_bsv8(const cldnn::memory& mem, std::stringstream & stream)
    {
        auto mem_arg = mem.get_layout();
        auto mem_ptr = mem.pointer<elemType>();

        auto i_size = mem_arg.size.batch[0]; //batch = input feature map
        auto x_size = mem_arg.size.spatial[0]; // spatial_x = output feature map
        auto weights_size = mem_arg.size.count();
        int xsv = 8, bsv = 8;
        unsigned int input_it = 0, input_i_it= 0 , input_o_it = 0;
        for (cldnn::tensor::value_type it = 0; it < weights_size; it++)
        {
                stream << convert_element(mem_ptr[input_it]) << " ";
                input_i_it++;
                if (input_i_it % bsv == 0) //separete every input slice with a new line
                {
                    stream << std::endl;
                    input_o_it++;
                    input_i_it = 0;
                }
                input_it = input_o_it*bsv + input_i_it;

                if (input_it % (xsv*bsv) == 0) // seperate every block (8x8) with a new line
                    stream << std::endl;
        }
    }

    template<typename elemType>
    void dump_bs_x_bsv16(const cldnn::memory& mem, std::stringstream & stream)
    {
        auto mem_arg = mem.get_layout();
        auto mem_ptr = mem.pointer<elemType>();

        auto i_size = mem_arg.size.batch[0]; //batch = input feature map
        auto x_size = mem_arg.size.spatial[0]; // spatial_x = output feature map
        auto weights_size = mem_arg.size.count();
        int bsv = 16;
        cldnn::tensor::value_type it = 0;
        while (it < weights_size)
        {
            stream << convert_element(mem_ptr[it]) << " ";
            it++;
            if (it % bsv == 0) //separate every bsv with a new line
                stream << std::endl;
        }
    }

    template <class T>
    void dump(const cldnn::memory& mem, std::stringstream& stream)
    {
        auto mem_ptr = mem.pointer<T>();

        auto&& pitches = mem.get_layout().get_pitches();
        auto&& size = mem.get_layout().size;
        for (cldnn::tensor::value_type b = 0; b < size.batch[0]; ++b)
        {
            stream << "============= BATCH " << b << " ============\n\n";
            for (cldnn::tensor::value_type f = 0; f < size.feature[0]; ++f)
            {
                stream << "feature " << f << ":\n";
                for (cldnn::tensor::value_type y = 0; y < size.spatial[1]; ++y)
                {
                    for (cldnn::tensor::value_type x = 0; x < size.spatial[0]; ++x)
                    {
                        unsigned int input_it = b*pitches.batch[0] + f*pitches.feature[0] + y*pitches.spatial[1] + x*pitches.spatial[0];
                        stream << convert_element(mem_ptr[input_it]) << " ";
                        input_it++;
                    }
                    stream << '\n';
                }
                stream << std::endl;
            }
        }
    }

    template <class T>
    void dump(const cldnn::memory& mem, std::vector<std::vector<std::string>>& dump_strings)
    {
        auto mem_ptr = mem.pointer<T>();
        std::stringstream stream;

        auto&& pitches = mem.get_layout().get_pitches();
        auto&& size = mem.get_layout().size;
        for (cldnn::tensor::value_type b = 0; b < size.batch[0]; ++b)
        {
            for (cldnn::tensor::value_type f = 0; f < size.feature[0]; ++f)
            {
                for (cldnn::tensor::value_type y = 0; y < size.spatial[1]; ++y)
                {
                    for (cldnn::tensor::value_type x = 0; x < size.spatial[0]; ++x)
                    {
                        unsigned int input_it = b*pitches.batch[0] + f*pitches.feature[0] + y*pitches.spatial[1] + x*pitches.spatial[0];
                        stream << convert_element(mem_ptr[input_it]) << " ";
                        input_it++;
                    }
                    stream << std::endl;
                    dump_strings[b][f] = stream.str();
                }
            }
        }
    }

    void logger::log_memory_to_file(const cldnn::memory& mem, std::string prefix, bool single_batch, cldnn::tensor::value_type batch_id, bool single_feature, cldnn::tensor::value_type feature_id)
    {
        auto batch = mem.get_layout().size.batch[0];
        auto feature = mem.get_layout().size.feature[0];
        auto eng_type =  "gpu" ;
        std::vector<std::vector<std::string>> dump_strings(batch);
        for(cldnn::tensor::value_type b = 0; b < batch; b++)
        {
            dump_strings[b].resize(feature);
        }

        if (mem.get_layout().data_type == cldnn::data_types::f32)
            dump<float>(mem, dump_strings);
        else
            dump<half_t>(mem, dump_strings);

        for (cldnn::tensor::value_type b = 0; b < batch; b++)
            for (cldnn::tensor::value_type f = 0; f < feature; f++)
            {
                if (!single_batch || (b == batch_id && f == feature_id))
                {
                    std::string filename((dump_dir + "/" + prefix + "_" + eng_type + "_b" + std::to_string(b) + "_f" + std::to_string(f) + ".txt"));
                    std::ofstream file_stream(filename);
                    file_stream << dump_strings[b][f];
                    file_stream.close();
                }
            }
    }

    void logger::log_weights_to_file(const cldnn::memory& mem, std::string prefix)
    {
        std::stringstream stream;

        if (mem.get_layout().data_type == cldnn::data_types::f32)
            dump<float>(mem, stream);
        else
            dump<half_t>(mem, stream);

        std::string filename((dump_dir + "/" + prefix + ".txt"));
        std::ofstream file_stream(filename);
        file_stream << stream.str();
        file_stream.close();
    }
}
