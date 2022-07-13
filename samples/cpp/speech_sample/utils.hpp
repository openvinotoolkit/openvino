// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cnpy.h>

#include <samples/common.hpp>

#define MAX_SCORE_DIFFERENCE 0.0001f  // max score difference for frame error threshold
#define MAX_VAL_2B_FEAT      16384    // max to find scale factor

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
typedef std::chrono::duration<float> fsec;

/**
 * @brief struct to store score error
 */
struct ScoreErrorT {
    uint32_t numScores;
    uint32_t numErrors;
    float threshold;
    float maxError;
    float rmsError;
    float sumError;
    float sumRmsError;
    float sumSquaredError;
    float maxRelError;
    float sumRelError;
    float sumSquaredRelError;
    float maxAbsRefScore;
    float sumAbsRefScore;
};

/**
 * @brief struct to store infer request data per frame
 */
struct InferRequestStruct {
    ov::InferRequest inferRequest;
    int frameIndex;
    uint32_t numFramesThisBatch;
};

/**
 * @brief Check number of input files and model network inputs
 * @param numInputs number model inputs
 * @param numInputFiles number of input files
 * @return none.
 */
void check_number_of_inputs(size_t numInputs, size_t numInputFiles) {
    if (numInputs != numInputFiles) {
        throw std::logic_error("Number of network inputs (" + std::to_string(numInputs) +
                               ")"
                               " is not equal to number of input files (" +
                               std::to_string(numInputFiles) + ")");
    }
}

/**
 * @brief Get scale factor for quantization
 * @param ptrFloatMemory pointer to float memory with speech feature vector
 * @param targetMax max scale factor
 * @param numElements number of elements in speech feature vector
 * @return scale factor
 */
float scale_factor_for_quantization(void* ptrFloatMemory, float targetMax, uint32_t numElements) {
    float* ptrFloatFeat = reinterpret_cast<float*>(ptrFloatMemory);
    float max = 0.0;
    float scaleFactor;

    for (uint32_t i = 0; i < numElements; i++) {
        if (fabs(ptrFloatFeat[i]) > max) {
            max = fabs(ptrFloatFeat[i]);
        }
    }

    if (max == 0) {
        scaleFactor = 1.0;
    } else {
        scaleFactor = targetMax / max;
    }

    return (scaleFactor);
}

/**
 * @brief Clean score error
 * @param error pointer to score error struct
 * @return none.
 */
void clear_score_error(ScoreErrorT* error) {
    error->numScores = 0;
    error->numErrors = 0;
    error->maxError = 0.0;
    error->rmsError = 0.0;
    error->sumError = 0.0;
    error->sumRmsError = 0.0;
    error->sumSquaredError = 0.0;
    error->maxRelError = 0.0;
    error->sumRelError = 0.0;
    error->sumSquaredRelError = 0.0;
    error->maxAbsRefScore = 0.0;
    error->sumAbsRefScore = 0.0;
}

/**
 * @brief Update total score error
 * @param error pointer to score error struct
 * @param totalError pointer to total score error struct
 * @return none.
 */
void update_score_error(ScoreErrorT* error, ScoreErrorT* totalError) {
    totalError->numErrors += error->numErrors;
    totalError->numScores += error->numScores;
    totalError->sumRmsError += error->rmsError;
    totalError->sumError += error->sumError;
    totalError->sumAbsRefScore += error->sumAbsRefScore;
    totalError->sumSquaredError += error->sumSquaredError;
    if (error->maxError > totalError->maxError) {
        totalError->maxError = error->maxError;
    }
    if (error->maxAbsRefScore > totalError->maxAbsRefScore) {
        totalError->maxAbsRefScore = error->maxAbsRefScore;
    }
    totalError->sumRelError += error->sumRelError;
    totalError->sumSquaredRelError += error->sumSquaredRelError;
    if (error->maxRelError > totalError->maxRelError) {
        totalError->maxRelError = error->maxRelError;
    }
}

/**
 * @brief Compare score errors, array should be the same length
 * @param ptrScoreArray - pointer to score error struct array
 * @param ptrRefScoreArray - pointer to score error struct array to compare
 * @param scoreError - pointer to score error struct to save a new error
 * @param numRows - number rows in score error arrays
 * @param numColumns - number columns in score error arrays
 * @return none.
 */
void compare_scores(float* ptrScoreArray,
                    void* ptrRefScoreArray,
                    ScoreErrorT* scoreError,
                    uint32_t numRows,
                    uint32_t numColumns) {
    uint32_t numErrors = 0;

    clear_score_error(scoreError);

    float* A = ptrScoreArray;
    float* B = reinterpret_cast<float*>(ptrRefScoreArray);
    for (uint32_t i = 0; i < numRows; i++) {
        for (uint32_t j = 0; j < numColumns; j++) {
            float score = A[i * numColumns + j];
            // std::cout << "score" << score << std::endl;
            float refscore = B[i * numColumns + j];
            float abs_refscore = fabs(refscore);
            float error = fabs(refscore - score);
            float rel_error = error / (static_cast<float>(abs_refscore) + 1e-20f);
            float squared_error = error * error;
            float squared_rel_error = rel_error * rel_error;
            scoreError->numScores++;
            scoreError->sumError += error;
            scoreError->sumAbsRefScore += abs_refscore;
            scoreError->sumSquaredError += squared_error;
            if (abs_refscore > scoreError->maxAbsRefScore) {
                scoreError->maxAbsRefScore = abs_refscore;
            }
            if (error > scoreError->maxError) {
                scoreError->maxError = error;
            }
            scoreError->sumRelError += rel_error;
            scoreError->sumSquaredRelError += squared_rel_error;
            if (rel_error > scoreError->maxRelError) {
                scoreError->maxRelError = rel_error;
            }
            if (error > scoreError->threshold) {
                numErrors++;
            }
        }
    }
    scoreError->rmsError = sqrt(scoreError->sumSquaredError / (numRows * numColumns));
    scoreError->sumRmsError += scoreError->rmsError;
    scoreError->numErrors = numErrors;
    // std::cout << "rmsError=" << scoreError->rmsError << "sumRmsError="<<scoreError->sumRmsError;
}

/**
 * @brief Get total stdev error
 * @param error pointer to score error struct
 * @return error
 */
float std_dev_error(ScoreErrorT error) {
    return (sqrt(error.sumSquaredError / error.numScores -
                 (error.sumError / error.numScores) * (error.sumError / error.numScores)));
}

#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
#    ifdef _WIN32
#        include <intrin.h>
#        include <windows.h>
#    else

#        include <cpuid.h>

#    endif

inline void native_cpuid(unsigned int* eax, unsigned int* ebx, unsigned int* ecx, unsigned int* edx) {
    size_t level = *eax;
#    ifdef _WIN32
    int regs[4] = {static_cast<int>(*eax), static_cast<int>(*ebx), static_cast<int>(*ecx), static_cast<int>(*edx)};
    __cpuid(regs, level);
    *eax = static_cast<uint32_t>(regs[0]);
    *ebx = static_cast<uint32_t>(regs[1]);
    *ecx = static_cast<uint32_t>(regs[2]);
    *edx = static_cast<uint32_t>(regs[3]);
#    else
    __get_cpuid(level, eax, ebx, ecx, edx);
#    endif
}

/**
 * @brief Get GNA module frequency
 * @return GNA module frequency in MHz
 */
float get_gna_frequency_mhz() {
    uint32_t eax = 1;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;
    uint32_t family = 0;
    uint32_t model = 0;
    const uint8_t sixth_family = 6;
    const uint8_t cannon_lake_model = 102;
    const uint8_t gemini_lake_model = 122;
    const uint8_t ice_lake_model = 126;
    const uint8_t tgl_model = 140;
    const uint8_t adl_s_model = 151;
    const uint8_t adl_p_model = 154;

    native_cpuid(&eax, &ebx, &ecx, &edx);
    family = (eax >> 8) & 0xF;

    // model is the concatenation of two fields
    // | extended model | model |
    // copy extended model data
    model = (eax >> 16) & 0xF;
    // shift
    model <<= 4;
    // copy model data
    model += (eax >> 4) & 0xF;

    if (family == sixth_family) {
        switch (model) {
        case cannon_lake_model:
        case ice_lake_model:
        case tgl_model:
        case adl_s_model:
        case adl_p_model:
            return 400;
        case gemini_lake_model:
            return 200;
        default:
            return 1;
        }
    } else {
        // counters not supported and we returns just default value
        return 1;
    }
}

#endif  // if not ARM

/**
 * @brief Print a report on the statistical score error
 * @param totalError reference to a total score error struct
 * @param framesNum number of frames in utterance
 * @param stream output stream
 * @return none.
 */
void print_reference_compare_results(ScoreErrorT const& totalError, size_t framesNum, std::ostream& stream) {
    stream << " max abs ref score: " << totalError.maxAbsRefScore << std::endl;
    stream << " avg abs ref score: " << totalError.sumAbsRefScore / totalError.numScores << std::endl;
    stream << "         max error: " << totalError.maxError << std::endl;
    stream << "         avg error: " << totalError.sumError / totalError.numScores << std::endl;
    stream << "     avg rms error: " << totalError.sumRmsError / framesNum << std::endl;
    stream << "       stdev error: " << std_dev_error(totalError) << std::endl << std::endl;
    stream << std::endl;
}

/**
 * @brief Print a report on the performance counts
 * @param utterancePerfMap reference to a map to store performance counters
 * @param numberOfFrames number of frames
 * @param stream output stream
 * @param fullDeviceName full device name string
 * @param numberOfFramesOnHw number of frames delivered to GNA HW
 * @param FLAGS_d flag of device
 * @return none.
 */
void print_performance_counters(std::map<std::string, ov::ProfilingInfo> const& utterancePerfMap,
                                size_t numberOfFrames,
                                std::ostream& stream,
                                std::string fullDeviceName,
                                const uint64_t numberOfFramesOnHw,
                                std::string FLAGS_d) {
#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
    std::ios::fmtflags fmt(std::cout.flags());
    stream << std::endl << "Performance counts:" << std::endl;
    stream << std::setw(10) << std::right << ""
           << "Counter descriptions";
    stream << std::setw(22) << "Utt scoring time";
    stream << std::setw(18) << "Avg infer time";
    stream << std::endl;

    stream << std::setw(46) << "(ms)";
    stream << std::setw(24) << "(us per call)";
    stream << std::endl;
    // if GNA HW counters
    // get frequency of GNA module
    float freq = get_gna_frequency_mhz();
    for (const auto& it : utterancePerfMap) {
        std::string const& counter_name = it.first;
        float current_units_us = static_cast<float>(it.second.real_time.count()) / freq;
        float call_units_us = 0;
        if (numberOfFrames == 0) {
            throw std::logic_error("Number off frames = 0,  division by zero.");
        } else {
            call_units_us = current_units_us / numberOfFrames;
        }
        if (FLAGS_d.find("GNA") != std::string::npos) {
            stream << std::setw(30) << std::left << counter_name.substr(4, counter_name.size() - 1);
        } else {
            stream << std::setw(30) << std::left << counter_name;
        }
        stream << std::setw(16) << std::right << current_units_us / 1000;
        stream << std::setw(21) << std::right << call_units_us;
        stream << std::endl;
    }
    stream << std::endl;
    std::cout << std::endl;
    std::cout << "Full device name: " << fullDeviceName << std::endl;
    std::cout << std::endl;
    stream << "Number of frames delivered to GNA HW: " << numberOfFramesOnHw;
    stream << "/" << numberOfFrames;
    stream << std::endl;
    std::cout.flags(fmt);
#endif
}

/**
 * @brief Get performance counts
 * @param request reference to infer request
 * @param perfCounters reference to a map to save performance counters
 * @return none.
 */
void get_performance_counters(ov::InferRequest& request, std::map<std::string, ov::ProfilingInfo>& perfCounters) {
    auto retPerfCounters = request.get_profiling_info();

    for (const auto& element : retPerfCounters) {
        perfCounters[element.node_name] = element;
    }
}

/**
 * @brief Summarize performance counts and total number of frames executed on the GNA HW device
 * @param perfCounters reference to a map to get performance counters
 * @param totalPerfCounters reference to a map to save total performance counters
 * @param totalRunsOnHw reference to a total number of frames computed on GNA HW
 * @return none.
 */
void sum_performance_counters(std::map<std::string, ov::ProfilingInfo> const& perfCounters,
                              std::map<std::string, ov::ProfilingInfo>& totalPerfCounters,
                              uint64_t& totalRunsOnHw) {
    auto runOnHw = false;
    for (const auto& pair : perfCounters) {
        totalPerfCounters[pair.first].real_time += pair.second.real_time;
        runOnHw |= pair.second.real_time > std::chrono::microseconds(0);  // if realTime is above zero, that means that
                                                                          // a primitive was executed on the device
    }
    totalRunsOnHw += runOnHw;
}

/**
 * @brief Split string by delimeter
 * @param s input string
 * @param delim delimeter
 * @return vector of chunks
 */
std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

/**
 * @brief Concat strings using delimeter
 * @param chunks input chunks
 * @param delim delimeter
 * @return concatenated string
 */
std::string concat(const std::vector<std::string>& chunks, char delim) {
    std::stringstream ss;
    for (auto&& chunk : chunks) {
        if (!ss.str().empty()) {
            ss << delim;
        }
        ss << chunk;
    }
    return ss.str();
}

/**
 * @brief Check whether name is present in node vector
 * @param nodes nodes
 * @param node_name name
 * @return false or true
 */
bool check_name(const ov::OutputVector& nodes, const std::string& node_name) {
    std::vector<std::string> any_names;
    bool count = false;
    for (auto& node : nodes) {
        any_names.push_back(node.get_any_name());
        auto names = node.get_names();
        count = std::count(names.begin(), names.end(), node_name);
        if (count)
            break;
    }
    if (!count) {
        std::stringstream ss;
        ss << "Incorrect node name '" + node_name << "'! ";
        ss << "Try one of the following names: [ ";
        for (auto&& name : any_names) {
            ss << name << " ";
        }
        ss << "]";
        throw std::logic_error(ss.str());
    }
    return count;
}

/**
 * @brief Parse scale factors per input
 * Format : <input_name1>=<sf1>,<input2>=<sf2> or just <sf>
 * @param inputs model inputs
 * @param values_string values_string input string
 * @return map of scale factors per input
 */
std::map<std::string, float> parse_scale_factors(const ov::OutputVector& inputs, const std::string& values_string) {
    auto get_sf = [&](const std::string& sf_string, const std::string& input_name = "") -> float {
        float sf;
        try {
            sf = std::stof(sf_string);
        } catch (...) {
            throw std::logic_error("Can't get float scale factor from: " + sf_string);
        }
        if (sf <= 0.0f) {
            throw std::logic_error("Scale factor for input '" + input_name +
                                   "' (counting from zero) is out of range (must be positive).");
        }
        return sf;
    };
    std::map<std::string, float> result;
    auto scale_factor_strings = split(values_string, ',');
    for (auto& scale_factor_string : scale_factor_strings) {
        auto values = split(scale_factor_string, '=');
        if (values.size() == 1) {
            if (scale_factor_strings.size() != 1) {
                throw std::logic_error("Unrecognized scale factor format! "
                                       "Please specify <input_name1>=<sf1>,<input_name2>=<sf2> or "
                                       "just <sf> to be applied to all inputs");
            }
            auto scale_factor = get_sf(values.at(0));
            for (auto& input : inputs) {
                result[input.get_any_name()] = scale_factor;
            }
        } else if (values.size() > 0) {
            auto sf_sting = values.back();
            values.pop_back();
            auto input_name = values.back();
            check_name(inputs, input_name);
            result[input_name] = get_sf(sf_sting, input_name);
        }
    }
    return result;
}

/**
 * @brief Parse string of file names separated by comma to save it to vector of file names
 * @param str file names separated by comma
 * @return vector of file names
 */
std::vector<std::string> convert_str_to_vector(std::string str) {
    std::vector<std::string> blobName;
    if (!str.empty()) {
        size_t pos_last = 0;
        size_t pos_next = 0;
        while ((pos_next = str.find(",", pos_last)) != std::string::npos) {
            blobName.push_back(str.substr(pos_last, pos_next - pos_last));
            pos_last = pos_next + 1;
        }
        blobName.push_back(str.substr(pos_last));
    }
    return blobName;
}

/**
 * @brief Parse layout string like "input0[value0],input1[value1]" or "[value]" (applied to all inputs)
 * @param layout_string input names with layout values
 * @param input_info reference to vector of inputs
 * @return map of inputs with layout values
 */
std::map<std::string, std::string> parse_input_layouts(const std::string& layout_string,
                                                       const std::vector<ov::Output<ov::Node>>& input_info) {
    // Parse parameter string like "input0[value0],input1[value1]" or "[value]" (applied to all
    // inputs)
    std::map<std::string, std::string> return_value;
    std::string search_string = layout_string;
    auto start_pos = search_string.find_first_of('[');
    auto input_name = search_string.substr(0, start_pos);
    while (start_pos != std::string::npos) {
        auto end_pos = search_string.find_first_of(']');
        if (end_pos == std::string::npos)
            break;
        if (start_pos)
            input_name = search_string.substr(0, start_pos);
        auto input_value = search_string.substr(start_pos + 1, end_pos - start_pos - 1);
        if (!input_name.empty()) {
            return_value[input_name] = input_value;
        } else {
            for (auto& item : input_info) {
                return_value[item.get_any_name()] = input_value;
            }
        }
        search_string = search_string.substr(end_pos + 1);
        if (search_string.empty() || (search_string.front() != ',' && search_string.front() != '['))
            break;
        if (search_string.front() == ',')
            search_string = search_string.substr(1);
        start_pos = search_string.find_first_of('[');
    }
    if (!search_string.empty())
        throw std::logic_error("Can't parse input parameter string: " + layout_string);
    return return_value;
}

/**
 * @brief Parse parameters for inputs/outputs/reference like as "<name1>=<file1.ark/.npz>,<name2>=<file2.ark/.npz>" or
 * "<file.ark/.npz>" in case of one input/output/reference.
 * @note Examplary result for given data: {"<file1.ark/.npz>,<file2.ark/.npz>",{"<name1>","<name2>"}}
 * @param file_paths_string input/output path
 * @return pair of filename and vector of layers names
 */
std::pair<std::string, std::vector<std::string>> parse_parameters(const std::string& file_paths_string) {
    auto search_string = file_paths_string;
    char comma_delim = ',';
    char equal_delim = '=';
    std::string filename = "";
    std::vector<std::string> layers_names;
    std::vector<std::string> filenames;
    if (!std::count(search_string.begin(), search_string.end(), comma_delim) &&
        !std::count(search_string.begin(), search_string.end(), equal_delim)) {
        return {search_string, layers_names};
    }
    search_string += comma_delim;
    std::vector<std::string> splitted = split(search_string, comma_delim);
    for (size_t j = 0; j < splitted.size(); j++) {
        auto equal_delim_pos = splitted[j].find_first_of(equal_delim);
        if (equal_delim_pos != std::string::npos) {
            layers_names.push_back(splitted[j].substr(0, equal_delim_pos));
            filenames.push_back(splitted[j].substr(equal_delim_pos + 1, std::string::npos));
        }
    }
    for (std::vector<std::string>::const_iterator name = filenames.begin(); name != filenames.end(); ++name) {
        filename += *name;
        if (name != filenames.end() - 1)
            filename += comma_delim;
    }
    return {filename, layers_names};
}