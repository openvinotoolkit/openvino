// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "inputs_filling.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "format_reader_ptr.h"
#include "samples/slog.hpp"
#include "utils.hpp"

template <typename T>
using uniformDistribution = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;

template <typename T>
ov::Tensor create_tensor_from_image(const std::vector<std::string>& files,
                                    size_t inputId,
                                    size_t batchSize,
                                    const benchmark_app::InputInfo& inputInfo,
                                    const std::string& inputName,
                                    std::string* filenames_used = nullptr) {
    auto tensor = ov::Tensor(inputInfo.type, inputInfo.dataShape);
    auto data = tensor.data<T>();

    /** Collect images data ptrs **/
    std::vector<std::shared_ptr<uint8_t>> vreader;
    vreader.reserve(batchSize);

    size_t imgBatchSize = 1;
    if (!inputInfo.layout.empty() && ov::layout::has_batch(inputInfo.layout)) {
        imgBatchSize = batchSize;
    } else {
        slog::warn << inputName << ": layout does not contain batch dimension. Assuming batch 1 for this input"
                   << slog::endl;
    }

    for (size_t b = 0; b < imgBatchSize; ++b) {
        auto inputIndex = (inputId + b) % files.size();
        if (filenames_used) {
            *filenames_used += (filenames_used->empty() ? "" : ", ") + files[inputIndex];
        }
        FormatReader::ReaderPtr reader(files[inputIndex].c_str());
        if (reader.get() == nullptr) {
            slog::warn << "Image " << files[inputIndex] << " cannot be read!" << slog::endl << slog::endl;
            continue;
        }

        /** Getting image data **/
        std::shared_ptr<uint8_t> imageData(reader->getData(inputInfo.width(), inputInfo.height()));
        if (imageData) {
            vreader.push_back(imageData);
        }
    }

    /** Fill input tensor with image. First b channel, then g and r channels **/
    const size_t numChannels = inputInfo.channels();
    const size_t width = inputInfo.width();
    const size_t height = inputInfo.height();
    /** Iterate over all input images **/
    for (size_t b = 0; b < imgBatchSize; ++b) {
        /** Iterate over all width **/
        for (size_t w = 0; w < width; ++w) {
            /** Iterate over all height **/
            for (size_t h = 0; h < height; ++h) {
                /** Iterate over all channels **/
                for (size_t ch = 0; ch < numChannels; ++ch) {
                    /**          [images stride + channels stride + pixel id ] all in
                     * bytes            **/
                    size_t offset = b * numChannels * width * height +
                                    (((inputInfo.layout == "NCHW") || (inputInfo.layout == "CHW"))
                                         ? (ch * width * height + h * width + w)
                                         : (h * width * numChannels + w * numChannels + ch));
                    data[offset] = static_cast<T>(vreader.at(b).get()[h * width * numChannels + w * numChannels + ch]);
                }
            }
        }
    }

    return tensor;
}

template <typename T>
ov::Tensor create_tensor_from_numpy(const std::vector<std::string>& files,
                                    size_t inputId,
                                    size_t batchSize,
                                    const benchmark_app::InputInfo& inputInfo,
                                    const std::string& inputName,
                                    std::string* filenames_used = nullptr) {
    size_t tensor_size =
        std::accumulate(inputInfo.dataShape.begin(), inputInfo.dataShape.end(), 1, std::multiplies<size_t>());
    auto tensor = ov::Tensor(inputInfo.type, inputInfo.dataShape);
    auto data = tensor.data<T>();

    std::vector<std::shared_ptr<unsigned char>> numpy_array_pointers;
    numpy_array_pointers.reserve(batchSize);

    size_t numpy_batch_size = 1;
    if (!inputInfo.layout.empty() && ov::layout::has_batch(inputInfo.layout)) {
        numpy_batch_size = batchSize;
    } else {
        slog::warn << inputName
                   << ": layout is not set or does not contain batch dimension. Assuming that numpy array "
                      "contains data for all batches."
                   << slog::endl;
    }

    tensor_size = tensor_size / numpy_batch_size;
    for (size_t b = 0; b < numpy_batch_size; ++b) {
        auto inputIndex = (inputId + b) % files.size();
        if (filenames_used) {
            *filenames_used += (filenames_used->empty() ? "" : ", ") + files[inputIndex];
        }
        FormatReader::ReaderPtr numpy_array_reader(files[inputIndex].c_str());
        if (numpy_array_reader.get() == nullptr) {
            slog::warn << "Numpy array " << files[inputIndex] << " cannot be read!" << slog::endl << slog::endl;
            continue;
        }

        std::shared_ptr<unsigned char> numpy_array_data_pointer(numpy_array_reader->getData());
        if (numpy_array_data_pointer) {
            numpy_array_pointers.push_back(numpy_array_data_pointer);
        }
    }

    size_t type_bytes_size = sizeof(T);
    std::unique_ptr<unsigned char[]> bytes_buffer(new unsigned char[type_bytes_size]);

    for (size_t batch_nr = 0; batch_nr < numpy_batch_size; ++batch_nr) {
        for (size_t input_tensor_nr = 0; input_tensor_nr < tensor_size; ++input_tensor_nr) {
            size_t offset = batch_nr * tensor_size + input_tensor_nr;
            for (size_t byte_nr = 0; byte_nr < type_bytes_size; ++byte_nr) {
                bytes_buffer.get()[byte_nr] =
                    numpy_array_pointers.at(batch_nr).get()[offset * type_bytes_size + byte_nr];
            }
            data[offset] = *((T*)(bytes_buffer.get()));
        }
    }

    return tensor;
}

template <typename T>
ov::Tensor create_tensor_im_info(const std::pair<size_t, size_t>& image_size,
                                 size_t batchSize,
                                 const benchmark_app::InputInfo& inputInfo,
                                 const std::string& inputName) {
    size_t tensor_size =
        std::accumulate(inputInfo.dataShape.begin(), inputInfo.dataShape.end(), 1, std::multiplies<size_t>());
    auto tensor = ov::Tensor(inputInfo.type, inputInfo.dataShape);
    char* data = static_cast<char*>(tensor.data());

    size_t infoBatchSize = 1;
    if (!inputInfo.layout.empty() && ov::layout::has_batch(inputInfo.layout)) {
        infoBatchSize = batchSize;
    } else {
        slog::warn << inputName << ": layout is not set or does not contain batch dimension. Assuming batch 1. "
                   << slog::endl;
    }

    for (size_t b = 0; b < infoBatchSize; b++) {
        size_t iminfoSize = tensor_size / infoBatchSize;
        for (size_t i = 0; i < iminfoSize; i++) {
            size_t index = b * iminfoSize + i;
            if (0 == i)
                data[index] = static_cast<char>(image_size.first);
            else if (1 == i)
                data[index] = static_cast<char>(image_size.second);
            else
                data[index] = static_cast<char>(1);
        }
    }

    return tensor;
}

template <typename T>
ov::Tensor create_tensor_from_binary(const std::vector<std::string>& files,
                                     size_t inputId,
                                     size_t batchSize,
                                     const benchmark_app::InputInfo& inputInfo,
                                     const std::string& inputName,
                                     std::string* filenames_used = nullptr) {
    size_t tensor_size =
        std::accumulate(inputInfo.dataShape.begin(), inputInfo.dataShape.end(), 1, std::multiplies<size_t>());
    auto tensor = ov::Tensor(inputInfo.type, inputInfo.dataShape);
    char* data = static_cast<char*>(tensor.data());
    size_t binaryBatchSize = 1;
    if (!inputInfo.layout.empty() && ov::layout::has_batch(inputInfo.layout)) {
        binaryBatchSize = batchSize;
    } else {
        slog::warn << inputName
                   << ": layout is not set or does not contain batch dimension. Assuming that binary "
                      "data read from file contains data for all batches."
                   << slog::endl;
    }

    for (size_t b = 0; b < binaryBatchSize; ++b) {
        size_t inputIndex = (inputId + b) % files.size();
        std::ifstream binaryFile(files[inputIndex], std::ios_base::binary | std::ios_base::ate);
        OPENVINO_ASSERT(binaryFile, "Cannot open ", files[inputIndex]);

        auto inputSize = tensor_size / binaryBatchSize;
        auto inputByteSize = inputSize * inputInfo.type.bitwidth() / 8u;
        std::string extension = get_extension(files[inputIndex]);
        if (extension == "bin") {
            auto fileSize = static_cast<std::size_t>(binaryFile.tellg());
            binaryFile.seekg(0, std::ios_base::beg);
            OPENVINO_ASSERT(binaryFile.good(), "Can not read ", files[inputIndex]);
            OPENVINO_ASSERT(fileSize == inputByteSize,
                            "File ",
                            files[inputIndex],
                            " contains ",
                            fileSize,
                            " bytes, but the model expects ",
                            inputByteSize);
        } else {
            OPENVINO_THROW("Unsupported binary file type: " + extension);
        }

        if (inputInfo.layout != "CN") {
            binaryFile.read(&data[b * inputByteSize], inputByteSize);
        } else {
            for (size_t i = 0; i < inputInfo.channels(); i++) {
                binaryFile.read(&data[(i * binaryBatchSize + b) * sizeof(T)], sizeof(T));
            }
        }

        if (filenames_used) {
            *filenames_used += (filenames_used->empty() ? "" : ", ") + files[inputIndex];
        }
    }

    return tensor;
}

template <typename T, typename T2>
ov::Tensor create_tensor_random(const benchmark_app::InputInfo& inputInfo,
                                T rand_min = std::numeric_limits<uint8_t>::min(),
                                T rand_max = std::numeric_limits<uint8_t>::max()) {
    size_t tensor_size =
        std::accumulate(inputInfo.dataShape.begin(), inputInfo.dataShape.end(), 1, std::multiplies<size_t>());
    auto tensor = ov::Tensor(inputInfo.type, inputInfo.dataShape);
    auto data = tensor.data<T>();

    std::mt19937 gen(0);
    uniformDistribution<T2> distribution(rand_min, rand_max);
    for (size_t i = 0; i < tensor_size; i++) {
        data[i] = static_cast<T>(distribution(gen));
    }

    return tensor;
}

ov::Tensor create_tensor_random_4bit(const benchmark_app::InputInfo& inputInfo,
                                     uint8_t rand_min = std::numeric_limits<uint8_t>::min(),
                                     uint8_t rand_max = std::numeric_limits<uint8_t>::max()) {
    auto tensor = ov::Tensor(inputInfo.type, inputInfo.dataShape);
    auto data = reinterpret_cast<uint8_t*>(tensor.data());

    std::mt19937 gen(0);
    uniformDistribution<int32_t> distribution(rand_min, rand_max);
    for (size_t i = 0; i < tensor.get_size(); i++) {
        uint8_t val = static_cast<uint8_t>(distribution(gen));
        size_t dst_idx = i / 2;
        if (i % 2) {
            data[dst_idx] = (data[dst_idx] & 0x0f) | (val << 4);
        } else {
            data[dst_idx] = (data[dst_idx] & 0xf0) | (val & 0x0f);
        }
    }

    return tensor;
}

ov::Tensor get_image_tensor(const std::vector<std::string>& files,
                            size_t inputId,
                            size_t batchSize,
                            const std::pair<std::string, benchmark_app::InputInfo>& inputInfo,
                            std::string* filenames_used = nullptr) {
    auto type = inputInfo.second.type;
    if (type == ov::element::f16) {
        return create_tensor_from_image<ov::float16>(files,
                                                     inputId,
                                                     batchSize,
                                                     inputInfo.second,
                                                     inputInfo.first,
                                                     filenames_used);
    } else if (type == ov::element::f32) {
        return create_tensor_from_image<float>(files,
                                               inputId,
                                               batchSize,
                                               inputInfo.second,
                                               inputInfo.first,
                                               filenames_used);
    } else if (type == ov::element::f64) {
        return create_tensor_from_image<double>(files,
                                                inputId,
                                                batchSize,
                                                inputInfo.second,
                                                inputInfo.first,
                                                filenames_used);
    } else if (type == ov::element::i8) {
        return create_tensor_from_image<int8_t>(files,
                                                inputId,
                                                batchSize,
                                                inputInfo.second,
                                                inputInfo.first,
                                                filenames_used);
    } else if (type == ov::element::i16) {
        return create_tensor_from_image<int16_t>(files,
                                                 inputId,
                                                 batchSize,
                                                 inputInfo.second,
                                                 inputInfo.first,
                                                 filenames_used);
    } else if (type == ov::element::i32) {
        return create_tensor_from_image<int32_t>(files,
                                                 inputId,
                                                 batchSize,
                                                 inputInfo.second,
                                                 inputInfo.first,
                                                 filenames_used);
    } else if (type == ov::element::i64) {
        return create_tensor_from_image<int64_t>(files,
                                                 inputId,
                                                 batchSize,
                                                 inputInfo.second,
                                                 inputInfo.first,
                                                 filenames_used);
    } else if ((type == ov::element::u8) || (type == ov::element::boolean)) {
        return create_tensor_from_image<uint8_t>(files,
                                                 inputId,
                                                 batchSize,
                                                 inputInfo.second,
                                                 inputInfo.first,
                                                 filenames_used);
    } else if (type == ov::element::u16) {
        return create_tensor_from_image<uint16_t>(files,
                                                  inputId,
                                                  batchSize,
                                                  inputInfo.second,
                                                  inputInfo.first,
                                                  filenames_used);
    } else if (type == ov::element::u32) {
        return create_tensor_from_image<uint32_t>(files,
                                                  inputId,
                                                  batchSize,
                                                  inputInfo.second,
                                                  inputInfo.first,
                                                  filenames_used);
    } else if (type == ov::element::u64) {
        return create_tensor_from_image<uint64_t>(files,
                                                  inputId,
                                                  batchSize,
                                                  inputInfo.second,
                                                  inputInfo.first,
                                                  filenames_used);
    } else {
        OPENVINO_THROW("Input type is not supported for " + inputInfo.first);
    }
}

ov::Tensor get_im_info_tensor(const std::pair<size_t, size_t>& image_size,
                              size_t batchSize,
                              const std::pair<std::string, benchmark_app::InputInfo>& inputInfo) {
    auto type = inputInfo.second.type;
    if (type == ov::element::f16) {
        return create_tensor_im_info<ov::float16>(image_size, batchSize, inputInfo.second, inputInfo.first);
    } else if (type == ov::element::f32) {
        return create_tensor_im_info<float>(image_size, batchSize, inputInfo.second, inputInfo.first);
    } else if (type == ov::element::f64) {
        return create_tensor_im_info<double>(image_size, batchSize, inputInfo.second, inputInfo.first);
    } else if (type == ov::element::i8) {
        return create_tensor_im_info<int8_t>(image_size, batchSize, inputInfo.second, inputInfo.first);
    } else if (type == ov::element::i16) {
        return create_tensor_im_info<int16_t>(image_size, batchSize, inputInfo.second, inputInfo.first);
    } else if (type == ov::element::i32) {
        return create_tensor_im_info<int32_t>(image_size, batchSize, inputInfo.second, inputInfo.first);
    } else if (type == ov::element::i64) {
        return create_tensor_im_info<int64_t>(image_size, batchSize, inputInfo.second, inputInfo.first);
    } else if ((type == ov::element::u8) || (type == ov::element::boolean)) {
        return create_tensor_im_info<uint8_t>(image_size, batchSize, inputInfo.second, inputInfo.first);
    } else if (type == ov::element::u16) {
        return create_tensor_im_info<uint16_t>(image_size, batchSize, inputInfo.second, inputInfo.first);
    } else if (type == ov::element::u32) {
        return create_tensor_im_info<uint32_t>(image_size, batchSize, inputInfo.second, inputInfo.first);
    } else if (type == ov::element::u64) {
        return create_tensor_im_info<uint64_t>(image_size, batchSize, inputInfo.second, inputInfo.first);
    } else {
        OPENVINO_THROW("Input type is not supported for " + inputInfo.first);
    }
}

ov::Tensor get_numpy_tensor(const std::vector<std::string>& files,
                            size_t inputId,
                            size_t batchSize,
                            const std::pair<std::string, benchmark_app::InputInfo>& inputInfo,
                            std::string* filenames_used = nullptr) {
    auto type = inputInfo.second.type;
    if (type == ov::element::f16) {
        return create_tensor_from_numpy<ov::float16>(files,
                                                     inputId,
                                                     batchSize,
                                                     inputInfo.second,
                                                     inputInfo.first,
                                                     filenames_used);
    } else if (type == ov::element::f32) {
        return create_tensor_from_numpy<float>(files,
                                               inputId,
                                               batchSize,
                                               inputInfo.second,
                                               inputInfo.first,
                                               filenames_used);
    } else if (type == ov::element::f64) {
        return create_tensor_from_numpy<double>(files,
                                                inputId,
                                                batchSize,
                                                inputInfo.second,
                                                inputInfo.first,
                                                filenames_used);
    } else if (type == ov::element::i8) {
        return create_tensor_from_numpy<int8_t>(files,
                                                inputId,
                                                batchSize,
                                                inputInfo.second,
                                                inputInfo.first,
                                                filenames_used);
    } else if (type == ov::element::i16) {
        return create_tensor_from_numpy<int16_t>(files,
                                                 inputId,
                                                 batchSize,
                                                 inputInfo.second,
                                                 inputInfo.first,
                                                 filenames_used);
    } else if (type == ov::element::i32) {
        return create_tensor_from_numpy<int32_t>(files,
                                                 inputId,
                                                 batchSize,
                                                 inputInfo.second,
                                                 inputInfo.first,
                                                 filenames_used);
    } else if (type == ov::element::i64) {
        return create_tensor_from_numpy<int64_t>(files,
                                                 inputId,
                                                 batchSize,
                                                 inputInfo.second,
                                                 inputInfo.first,
                                                 filenames_used);
    } else if ((type == ov::element::u8) || (type == ov::element::boolean)) {
        return create_tensor_from_numpy<uint8_t>(files,
                                                 inputId,
                                                 batchSize,
                                                 inputInfo.second,
                                                 inputInfo.first,
                                                 filenames_used);
    } else if (type == ov::element::u16) {
        return create_tensor_from_numpy<uint16_t>(files,
                                                  inputId,
                                                  batchSize,
                                                  inputInfo.second,
                                                  inputInfo.first,
                                                  filenames_used);
    } else if (type == ov::element::u32) {
        return create_tensor_from_numpy<uint32_t>(files,
                                                  inputId,
                                                  batchSize,
                                                  inputInfo.second,
                                                  inputInfo.first,
                                                  filenames_used);
    } else if (type == ov::element::u64) {
        return create_tensor_from_numpy<uint64_t>(files,
                                                  inputId,
                                                  batchSize,
                                                  inputInfo.second,
                                                  inputInfo.first,
                                                  filenames_used);
    } else {
        OPENVINO_THROW("Input type is not supported for " + inputInfo.first);
    }
}

ov::Tensor get_binary_tensor(const std::vector<std::string>& files,
                             size_t inputId,
                             size_t batchSize,
                             const std::pair<std::string, benchmark_app::InputInfo>& inputInfo,
                             std::string* filenames_used = nullptr) {
    const auto& type = inputInfo.second.type;
    if (type == ov::element::f16) {
        return create_tensor_from_binary<ov::float16>(files,
                                                      inputId,
                                                      batchSize,
                                                      inputInfo.second,
                                                      inputInfo.first,
                                                      filenames_used);
    } else if (type == ov::element::f32) {
        return create_tensor_from_binary<float>(files,
                                                inputId,
                                                batchSize,
                                                inputInfo.second,
                                                inputInfo.first,
                                                filenames_used);
    } else if (type == ov::element::f64) {
        return create_tensor_from_binary<double>(files,
                                                 inputId,
                                                 batchSize,
                                                 inputInfo.second,
                                                 inputInfo.first,
                                                 filenames_used);
    } else if (type == ov::element::i8 || (type == ov::element::i4)) {
        return create_tensor_from_binary<int8_t>(files,
                                                 inputId,
                                                 batchSize,
                                                 inputInfo.second,
                                                 inputInfo.first,
                                                 filenames_used);
    } else if (type == ov::element::i16) {
        return create_tensor_from_binary<int16_t>(files,
                                                  inputId,
                                                  batchSize,
                                                  inputInfo.second,
                                                  inputInfo.first,
                                                  filenames_used);
    } else if (type == ov::element::i32) {
        return create_tensor_from_binary<int32_t>(files,
                                                  inputId,
                                                  batchSize,
                                                  inputInfo.second,
                                                  inputInfo.first,
                                                  filenames_used);
    } else if (type == ov::element::i64) {
        return create_tensor_from_binary<int64_t>(files,
                                                  inputId,
                                                  batchSize,
                                                  inputInfo.second,
                                                  inputInfo.first,
                                                  filenames_used);
    } else if ((type == ov::element::u8) || (type == ov::element::boolean) || (type == ov::element::u4)) {
        return create_tensor_from_binary<uint8_t>(files,
                                                  inputId,
                                                  batchSize,
                                                  inputInfo.second,
                                                  inputInfo.first,
                                                  filenames_used);
    } else if (type == ov::element::u16) {
        return create_tensor_from_binary<uint16_t>(files,
                                                   inputId,
                                                   batchSize,
                                                   inputInfo.second,
                                                   inputInfo.first,
                                                   filenames_used);
    } else if (type == ov::element::u32) {
        return create_tensor_from_binary<uint32_t>(files,
                                                   inputId,
                                                   batchSize,
                                                   inputInfo.second,
                                                   inputInfo.first,
                                                   filenames_used);
    } else if (type == ov::element::u64) {
        return create_tensor_from_binary<uint64_t>(files,
                                                   inputId,
                                                   batchSize,
                                                   inputInfo.second,
                                                   inputInfo.first,
                                                   filenames_used);
    } else {
        OPENVINO_THROW("Input type is not supported for " + inputInfo.first);
    }
}

ov::Tensor get_random_tensor(const std::pair<std::string, benchmark_app::InputInfo>& inputInfo) {
    auto type = inputInfo.second.type;
    if (type == ov::element::f32) {
        return create_tensor_random<float, float>(inputInfo.second);
    } else if (type == ov::element::f64) {
        return create_tensor_random<double, double>(inputInfo.second);
    } else if (type == ov::element::f16) {
        return create_tensor_random<ov::float16, float>(inputInfo.second);
    } else if (type == ov::element::i32) {
        return create_tensor_random<int32_t, int32_t>(inputInfo.second);
    } else if (type == ov::element::i64) {
        return create_tensor_random<int64_t, int64_t>(inputInfo.second);
    } else if ((type == ov::element::u8) || (type == ov::element::boolean)) {
        // uniform_int_distribution<uint8_t> is not allowed in the C++17
        // standard and vs2017/19
        return create_tensor_random<uint8_t, uint32_t>(inputInfo.second);
    } else if (type == ov::element::i8) {
        // uniform_int_distribution<int8_t> is not allowed in the C++17 standard
        // and vs2017/19
        return create_tensor_random<int8_t, int32_t>(inputInfo.second,
                                                     std::numeric_limits<int8_t>::min(),
                                                     std::numeric_limits<int8_t>::max());
    } else if (type == ov::element::u16) {
        return create_tensor_random<uint16_t, uint16_t>(inputInfo.second);
    } else if (type == ov::element::i16) {
        return create_tensor_random<int16_t, int16_t>(inputInfo.second);
    } else if (type == ov::element::boolean) {
        return create_tensor_random<uint8_t, uint32_t>(inputInfo.second, 0, 1);
    } else if (type == ov::element::u4) {
        return create_tensor_random_4bit(inputInfo.second, 0, 15);
    } else if (type == ov::element::i4) {
        return create_tensor_random_4bit(inputInfo.second, 0, 15);
    } else if (type == ov::element::string) {
        const auto& in_info = inputInfo.second;
        const auto tensor_size = ov::shape_size(in_info.dataShape);
        auto tensor = ov::Tensor(in_info.type, in_info.dataShape);
        auto data = tensor.data<std::string>();

        std::mt19937 str_len_gen(0);
        uniformDistribution<uint32_t> len_distribution(20, 50);
        std::mt19937 char_val_gen(0);
        uniformDistribution<uint32_t> char_distribution(0, 127);
        for (size_t i = 0; i < tensor_size; i++) {
            data[i].resize(len_distribution(str_len_gen));
            for (size_t j = 0lu; j < data[i].size(); j++) {
                data[i][j] = static_cast<char>(char_distribution(char_val_gen));
            }
        }

        return tensor;
    } else {
        OPENVINO_THROW("Input type is not supported for " + inputInfo.first);
    }
}

std::string get_test_info_stream_header(benchmark_app::InputInfo& inputInfo) {
    std::stringstream strOut;
    strOut << "(" << inputInfo.layout.to_string() << ", " << inputInfo.type.get_type_name() << ", "
           << inputInfo.dataShape << ", ";
    if (inputInfo.partialShape.is_dynamic()) {
        strOut << std::string("dyn:") << inputInfo.partialShape << "):\t";
    } else {
        strOut << "static):\t";
    }
    return strOut.str();
}

std::map<std::string, ov::TensorVector> get_tensors(std::map<std::string, std::vector<std::string>> inputFiles,
                                                    std::vector<benchmark_app::InputsInfo>& app_inputs_info) {
    std::ios::fmtflags fmt(std::cout.flags());
    std::map<std::string, ov::TensorVector> tensors;
    if (app_inputs_info.empty()) {
        throw std::logic_error("Inputs Info for model is empty!");
    }

    if (!inputFiles.empty() && inputFiles.size() != app_inputs_info[0].size()) {
        throw std::logic_error("Number of inputs specified in -i must be equal to number of model inputs!");
    }

    // count image type inputs of network
    std::vector<std::pair<size_t, size_t>> net_input_im_sizes;
    for (auto& inputs_info : app_inputs_info) {
        for (auto& input : inputs_info) {
            if (input.second.is_image()) {
                net_input_im_sizes.push_back(std::make_pair(input.second.width(), input.second.height()));
            }
        }
    }

    for (auto& files : inputFiles) {
        if (!files.first.empty() && app_inputs_info[0].find(files.first) == app_inputs_info[0].end()) {
            throw std::logic_error("Input name \"" + files.first +
                                   "\" used in -i parameter doesn't match any model's input");
        }

        std::string input_name = files.first.empty() ? app_inputs_info[0].begin()->first : files.first;
        auto input = app_inputs_info[0].at(input_name);
        if (!files.second.empty() && files.second[0] != "random" && files.second[0] != "image_info") {
            auto filtered_numpy_files = filter_files_by_extensions(files.second, supported_numpy_extensions);
            auto filtered_image_files = filter_files_by_extensions(files.second, supported_image_extensions);

            if (!filtered_numpy_files.empty()) {
                files.second = filtered_numpy_files;
            } else if (!filtered_image_files.empty() && input.is_image()) {
                files.second = filtered_image_files;
            } else if (input.is_image_info() && net_input_im_sizes.size() == app_inputs_info.size()) {
                slog::info << "Input '" << input_name
                           << "' probably is image info. All files for this input will"
                              " be ignored."
                           << slog::endl;
                files.second = {"image_info"};
                continue;
            } else {
                files.second = filter_files_by_extensions(files.second, supported_binary_extensions);
            }
        }

        if (files.second.empty()) {
            slog::warn << "No suitable files for input were found! Random data will be used for input " << input_name
                       << slog::endl;
            files.second = {"random"};
        }

        size_t filesToBeUsed = 0;
        size_t shapesToBeUsed = 0;
        if (files.second.size() > app_inputs_info.size()) {
            shapesToBeUsed = app_inputs_info.size();
            filesToBeUsed = files.second.size() - files.second.size() % app_inputs_info.size();
            if (filesToBeUsed != files.second.size()) {
                slog::warn << "Number of files must be a multiple of the number of shapes for certain input. Only " +
                                  std::to_string(filesToBeUsed) + " files will be added."
                           << slog::endl;
            }
            while (files.second.size() != filesToBeUsed) {
                files.second.pop_back();
            }
        } else {
            shapesToBeUsed = app_inputs_info.size() - app_inputs_info.size() % files.second.size();
            filesToBeUsed = files.second.size();
            if (shapesToBeUsed != app_inputs_info.size()) {
                slog::warn << "Number of data shapes must be a multiple of the number of files. For input "
                           << files.first << " only " + std::to_string(shapesToBeUsed) + " files will be added."
                           << slog::endl;
            }
            while (app_inputs_info.size() != shapesToBeUsed) {
                app_inputs_info.pop_back();
                net_input_im_sizes.pop_back();
            }
        }
    }

    std::vector<std::map<std::string, std::string>> logOutput;
    // All inputs should process equal number of files, so for the case of N, 1, N number of files,
    // second input also should have N blobs cloned from 1 file
    size_t filesNum = 0;
    if (!inputFiles.empty()) {
        filesNum = std::max_element(inputFiles.begin(),
                                    inputFiles.end(),
                                    [](const std::pair<std::string, std::vector<std::string>>& a,
                                       const std::pair<std::string, std::vector<std::string>>& b) {
                                        return a.second.size() < b.second.size();
                                    })
                       ->second.size();
    } else {
        std::vector<std::pair<size_t, size_t>> net_input_im_sizes;
        for (auto& input_info : app_inputs_info[0]) {
            inputFiles[input_info.first] = {"random"};
        }
    }

    std::vector<size_t> batchSizes;
    for (const auto& info : app_inputs_info) {
        batchSizes.push_back(get_batch_size(info));
    }

    for (const auto& files : inputFiles) {
        std::string input_name = files.first.empty() ? app_inputs_info[0].begin()->first : files.first;
        size_t n_shape = 0, m_file = 0;
        while (n_shape < app_inputs_info.size() || m_file < filesNum) {
            size_t batchSize = batchSizes[n_shape % app_inputs_info.size()];
            size_t inputId = m_file % files.second.size();
            auto input_info = app_inputs_info[n_shape % app_inputs_info.size()].at(input_name);

            std::string tensor_src_info;
            if (files.second[0] == "random") {
                // Fill random
                tensor_src_info = "random (" +
                                  std::string((input_info.is_image() ? "image/numpy array" : "binary data")) +
                                  " is expected)";
                tensors[input_name].push_back(get_random_tensor({input_name, input_info}));
            } else if (files.second[0] == "image_info") {
                // Most likely it is image info: fill with image information
                auto image_size = net_input_im_sizes.at(n_shape % app_inputs_info.size());
                tensor_src_info =
                    "Image size tensor " + std::to_string(image_size.first) + " x " + std::to_string(image_size.second);
                tensors[input_name].push_back(get_im_info_tensor(image_size, batchSize, {input_name, input_info}));
            } else if (supported_numpy_extensions.count(get_extension(files.second[0]))) {
                // Fill with Numpy arrrays
                tensors[input_name].push_back(
                    get_numpy_tensor(files.second, inputId, batchSize, {input_name, input_info}, &tensor_src_info));
            } else if (input_info.is_image()) {
                // Fill with Images
                tensors[input_name].push_back(
                    get_image_tensor(files.second, inputId, batchSize, {input_name, input_info}, &tensor_src_info));
            } else {
                // Fill with binary files
                tensors[input_name].push_back(
                    get_binary_tensor(files.second, inputId, batchSize, {input_name, input_info}, &tensor_src_info));
            }

            // Preparing info
            std::string strOut = get_test_info_stream_header(input_info) + tensor_src_info;
            if (n_shape >= logOutput.size()) {
                logOutput.resize(n_shape + 1);
            }
            logOutput[n_shape][input_name] += strOut;

            ++n_shape;
            m_file += batchSize;
        }
    }

    for (size_t i = 0; i < logOutput.size(); i++) {
        slog::info << "Test Config " << i << slog::endl;
        auto maxNameWidth = std::max_element(logOutput[i].begin(),
                                             logOutput[i].end(),
                                             [](const std::pair<std::string, std::string>& a,
                                                const std::pair<std::string, std::string>& b) {
                                                 return a.first.size() < b.first.size();
                                             })
                                ->first.size();
        for (const std::pair<const std::string, std::string>& inputLog : logOutput[i]) {
            slog::info << std::left << std::setw(maxNameWidth + 2) << inputLog.first << inputLog.second << slog::endl;
        }
    }
    std::cout.flags(fmt);

    return tensors;
}

std::map<std::string, ov::TensorVector> get_tensors_static_case(const std::vector<std::string>& inputFiles,
                                                                const size_t& batchSize,
                                                                benchmark_app::InputsInfo& app_inputs_info,
                                                                size_t requestsNum) {
    std::ios::fmtflags fmt(std::cout.flags());
    std::map<std::string, ov::TensorVector> blobs;

    std::vector<std::pair<size_t, size_t>> net_input_im_sizes;
    for (auto& item : app_inputs_info) {
        if (item.second.partialShape.is_static() && item.second.is_image()) {
            net_input_im_sizes.push_back(std::make_pair(item.second.width(), item.second.height()));
        }
    }

    std::vector<std::string> binaryFiles = filter_files_by_extensions(inputFiles, supported_binary_extensions);
    std::vector<std::string> numpyFiles = filter_files_by_extensions(inputFiles, supported_numpy_extensions);
    std::vector<std::string> imageFiles = filter_files_by_extensions(inputFiles, supported_image_extensions);

    size_t imageInputsNum = imageFiles.size();
    size_t numpyInputsNum = numpyFiles.size();
    size_t binaryInputsNum = binaryFiles.size();
    size_t totalInputsNum = imageInputsNum + numpyInputsNum + binaryInputsNum;

    if (inputFiles.empty()) {
        slog::warn << "No input files were given: all inputs will be filled with "
                      "random values!"
                   << slog::endl;
    } else {
        std::sort(std::begin(binaryFiles), std::end(binaryFiles));
        std::sort(std::begin(numpyFiles), std::end(numpyFiles));
        std::sort(std::begin(imageFiles), std::end(imageFiles));

        auto filesToBeUsed = totalInputsNum * batchSize * requestsNum;
        if (filesToBeUsed == 0 && !inputFiles.empty()) {
            std::stringstream ss;
            for (auto& ext : supported_image_extensions) {
                if (!ss.str().empty()) {
                    ss << ", ";
                }
                ss << ext;
            }
            for (auto& ext : supported_numpy_extensions) {
                if (!ss.str().empty()) {
                    ss << ", ";
                }
                ss << ext;
            }
            for (auto& ext : supported_binary_extensions) {
                if (!ss.str().empty()) {
                    ss << ", ";
                }
                ss << ext;
            }
            slog::warn << "Inputs of unsupported type found! Please check your file "
                          "extensions: "
                       << ss.str() << slog::endl;
        } else if (app_inputs_info.size() > totalInputsNum) {
            slog::warn << "Some input files will be duplicated: " << filesToBeUsed << " files are required but only "
                       << totalInputsNum << " are provided" << slog::endl;
        } else if (filesToBeUsed < app_inputs_info.size()) {
            slog::warn << "Some input files will be ignored: only " << filesToBeUsed << " are required from "
                       << totalInputsNum << slog::endl;
        }
    }

    std::map<std::string, std::vector<std::string>> mappedFiles;
    size_t imageInputsCount = 0;
    size_t numpyInputsCount = 0;
    size_t binaryInputsCount = 0;
    for (auto& input : app_inputs_info) {
        if (numpyInputsNum) {
            mappedFiles[input.first] = {};
            for (size_t i = 0; i < numpyFiles.size(); i += numpyInputsNum) {
                mappedFiles[input.first].push_back(
                    numpyFiles[(numpyInputsCount + i) * numpyInputsNum % numpyFiles.size()]);
            }
            ++numpyInputsCount;
        } else if (input.second.is_image()) {
            mappedFiles[input.first] = {};
            for (size_t i = 0; i < imageFiles.size(); i += imageInputsNum) {
                mappedFiles[input.first].push_back(
                    imageFiles[(imageInputsCount + i) * imageInputsNum % imageFiles.size()]);
            }
            ++imageInputsCount;
        } else {
            mappedFiles[input.first] = {};
            if (!binaryFiles.empty()) {
                for (size_t i = 0; i < binaryFiles.size(); i += binaryInputsNum) {
                    mappedFiles[input.first].push_back(binaryFiles[(binaryInputsCount + i) % binaryFiles.size()]);
                }
            }
            ++binaryInputsCount;
        }
    }

    size_t filesNum = 0;
    if (!inputFiles.empty()) {
        filesNum = std::max_element(mappedFiles.begin(),
                                    mappedFiles.end(),
                                    [](const std::pair<std::string, std::vector<std::string>>& a,
                                       const std::pair<std::string, std::vector<std::string>>& b) {
                                        return a.second.size() < b.second.size();
                                    })
                       ->second.size();
    }
    size_t test_configs_num = filesNum / batchSize == 0 ? 1 : filesNum / batchSize;
    std::vector<std::map<std::string, std::string>> logOutput(test_configs_num);
    for (const auto& files : mappedFiles) {
        size_t imageInputId = 0;
        size_t numpyInputId = 0;
        size_t binaryInputId = 0;
        auto input_name = files.first;
        auto input_info = app_inputs_info.at(files.first);

        for (size_t i = 0; i < test_configs_num; ++i) {
            std::string blob_src_info;
            if (files.second.size() && supported_numpy_extensions.count(get_extension(files.second[0]))) {
                if (!numpyFiles.empty()) {
                    // Fill with Numpy arryys
                    blobs[input_name].push_back(get_numpy_tensor(files.second,
                                                                 imageInputId,
                                                                 batchSize,
                                                                 {input_name, input_info},
                                                                 &blob_src_info));
                    numpyInputId = (numpyInputId + batchSize) % files.second.size();
                    logOutput[i][input_name] += get_test_info_stream_header(input_info) + blob_src_info;
                    continue;
                }
            } else if (input_info.is_image()) {
                if (!imageFiles.empty()) {
                    // Fill with Images
                    blobs[input_name].push_back(get_image_tensor(files.second,
                                                                 imageInputId,
                                                                 batchSize,
                                                                 {input_name, input_info},
                                                                 &blob_src_info));
                    imageInputId = (imageInputId + batchSize) % files.second.size();
                    logOutput[i][input_name] += get_test_info_stream_header(input_info) + blob_src_info;
                    continue;
                }
            } else {
                if (!binaryFiles.empty()) {
                    // Fill with binary files
                    blobs[input_name].push_back(get_binary_tensor(files.second,
                                                                  binaryInputId,
                                                                  batchSize,
                                                                  {input_name, input_info},
                                                                  &blob_src_info));
                    binaryInputId = (binaryInputId + batchSize) % files.second.size();
                    logOutput[i][input_name] += get_test_info_stream_header(input_info) + blob_src_info;
                    continue;
                }
                if (input_info.is_image_info() && (net_input_im_sizes.size() == 1)) {
                    // Most likely it is image info: fill with image information
                    auto image_size = net_input_im_sizes.at(0);
                    blob_src_info = "Image size blob " + std::to_string(image_size.first) + " x " +
                                    std::to_string(image_size.second);
                    blobs[input_name].push_back(get_im_info_tensor(image_size, batchSize, {input_name, input_info}));
                    logOutput[i][input_name] += get_test_info_stream_header(input_info) + blob_src_info;
                    continue;
                }
            }
            // Fill random
            blob_src_info = "random (" + std::string((input_info.is_image() ? "image" : "binary data")) +
                            "/numpy array is expected)";
            blobs[input_name].push_back(get_random_tensor({input_name, input_info}));
            logOutput[i][input_name] += get_test_info_stream_header(input_info) + blob_src_info;
        }
    }

    for (size_t i = 0; i < logOutput.size(); i++) {
        slog::info << "Test Config " << i << slog::endl;
        auto maxNameWidth = std::max_element(logOutput[i].begin(),
                                             logOutput[i].end(),
                                             [](const std::pair<std::string, std::string>& a,
                                                const std::pair<std::string, std::string>& b) {
                                                 return a.first.size() < b.first.size();
                                             })
                                ->first.size();
        for (auto inputLog : logOutput[i]) {
            slog::info << std::left << std::setw(maxNameWidth + 2) << inputLog.first << inputLog.second << slog::endl;
        }
    }
    std::cout.flags(fmt);

    return blobs;
}

void copy_tensor_data(ov::Tensor& dst, const ov::Tensor& src) {
    if (src.get_shape() != dst.get_shape() || src.get_byte_size() != dst.get_byte_size()) {
        throw std::runtime_error(
            "Source and destination tensors shapes and byte sizes are expected to be equal for data copying.");
    }

    memcpy(dst.data(), src.data(), src.get_byte_size());
}
