// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_transpose.hpp"
#include "ie_parallel.hpp"
#include "nodes/common/cpu_memcpy.h"
#include "mlas.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

static inline size_t parallel_init(size_t start, size_t nDims, const SizeVector& dims, SizeVector& indexes) {
    for (int j = nDims - 1; j >= 0; j--) {
        indexes[j] = start % dims[j];
        start = start / dims[j];
    }
    return start;
}

static inline void parallel_step(size_t nDims, const SizeVector& dims, SizeVector& indexes) {
    for (int j = nDims - 1; j >= 0; --j) {
        ++indexes[j];
        if (indexes[j] < dims[j])
            break;
        else
            indexes[j] = 0;
    }
}

template <typename T>
struct has_mlas_transpose : std::false_type {};

template <>
struct has_mlas_transpose<uint8_t> : std::true_type {};

template <>
struct has_mlas_transpose<uint32_t> : std::true_type {};

template <typename T>
typename std::enable_if<!has_mlas_transpose<T>::value, void>::type SimpleTransposeSingleAxisOutwards(
    const T* input_data, T* output_data, int64_t num_loops, int64_t num_writers, int64_t writes_per_loop, int64_t writes_per_writer_per_loop) {
  const T* end;
  //std::cout << "SimpleTransposeSingleAxisOutwards - non-MLAS branch" << std::endl;
  for (int64_t l = 0; l < num_loops; ++l) {
    T* output_for_first_writer = output_data;
    for (auto wwpl = 0; wwpl < writes_per_writer_per_loop; ++wwpl) {
      T* output_for_current_writer = output_for_first_writer;
      end = input_data + num_writers;
      for (; input_data != end;) {
        *output_for_current_writer = *input_data++;
        // skip to output position for next writer
        output_for_current_writer += writes_per_writer_per_loop;
      }
      ++output_for_first_writer;
    }
    output_data += writes_per_loop;
  }
}

template <typename T>
typename std::enable_if<has_mlas_transpose<T>::value, void>::type SimpleTransposeSingleAxisOutwards(
    const T* input_data, T* output_data, int64_t num_loops, int64_t num_writers, int64_t writes_per_loop, int64_t writes_per_writer_per_loop) {
  //std::cout << "SimpleTransposeSingleAxisOutwards - MLAS branch" << std::endl;
  for (int64_t l = 0; l < num_loops; ++l) {
    MlasTranspose(input_data, output_data, static_cast<size_t>(writes_per_writer_per_loop), static_cast<size_t>(num_writers));
    input_data += writes_per_loop;
    output_data += writes_per_loop;
  }
}

template <typename T>
typename std::enable_if<!has_mlas_transpose<T>::value, void>::type SimpleTransposeSingleAxisInwards(
    const T* input_data, T* output_data, int64_t num_loops, int64_t num_readers, int64_t reads_per_loop, int64_t reads_per_reader_per_loop) {
  T* end;
  //std::cout << "SimpleTransposeSingleAxisInwards - non-MLAS branch" << std::endl;
  for (int64_t l = 0; l < num_loops; ++l) {
    const T* input_for_first_reader = input_data;
    for (auto rrpl = 0; rrpl < reads_per_reader_per_loop; ++rrpl) {
      const T* input_for_current_reader = input_for_first_reader;
      end = output_data + num_readers;
      for (; output_data != end;) {
        *output_data++ = *input_for_current_reader;
        // skip to input position for next reader
        input_for_current_reader += reads_per_reader_per_loop;
      }
      ++input_for_first_reader;
    }
    input_data += reads_per_loop;
  }
}

template <typename T>
typename std::enable_if<has_mlas_transpose<T>::value, void>::type SimpleTransposeSingleAxisInwards(
    const T* input_data, T* output_data, int64_t num_loops, int64_t num_readers, int64_t reads_per_loop, int64_t reads_per_reader_per_loop) {
  //std::cout << "SimpleTransposeSingleAxisInwards - MLAS branch" << std::endl;
  for (int64_t l = 0; l < num_loops; ++l) {
    MlasTranspose(input_data, output_data, static_cast<size_t>(num_readers), static_cast<size_t>(reads_per_reader_per_loop));
    input_data += reads_per_loop;
    output_data += reads_per_loop;
  }
}

bool RefTransposeExecutor::IsTransposeMovingSingleAxis(SizeVector permutations, size_t& from, size_t& to) {
  // if a single axis moved to an outer dimension, the values should be one lower than the index until the slot the
  // axis was moved from, and equal to the index after that.
  // e.g. axis 3 moves out to 1 would be: 0, 3, 1, 2, 4
  auto check_moved_outwards = [&permutations](size_t cur, size_t moved_from) {
    // we start processing with the slot after the moved one, so the expected value is one less than the index
    size_t expected = cur - 1;
    for (size_t end = permutations.size(); cur < end; ++cur) {
      if (permutations[cur] != expected) {
        return false;
      }
      // we are at the slot the axis moved from, so do an additional increment before checking the next value
      if (cur == moved_from) {
        ++expected;
      }
      ++expected;
    }
    return true;
  };
  // if a single axis moved to an inner dimension, the values should be one higher than the index until the slot the
  // axis was moved to, and equal to the index after that.
  // e.g. axis 1 moves inwards to 3 would be: 0, 2, 3, 1, 4
  auto check_moved_inwards = [&permutations](size_t cur, size_t& moved_to) {
    size_t started_at = cur;
    size_t expected = cur + 1;
    moved_to = std::numeric_limits<size_t>::max();
    for (size_t end = permutations.size(); cur < end; ++cur) {
      if (permutations[cur] != expected) {
        // if a single axis moved it must have come from the location we started at
        if (started_at != permutations[cur]) {
          return false;
        }
        moved_to = cur;
      } else {
        ++expected;
      }
    }
    return moved_to != std::numeric_limits<size_t>::max();
  };
  bool single_axis_moved = false;
  // check axis moving outwards (earlier entry in permutations)
  for (size_t i = 0, end = permutations.size(); i < end; ++i) {
    size_t axis = permutations[i];
    if (axis != i) {
      if (check_moved_outwards(i + 1, axis)) {
        single_axis_moved = true;
        to = i;
        from = axis;
      } else if (check_moved_inwards(i, to)) {
        single_axis_moved = true;
        from = i;
      }
      break;
    }
  }
  return single_axis_moved;
}

void RefTransposeExecutor::TransposeSingleAxisOutwards(const MemoryCPtr& input, const MemoryPtr& output, size_t from, size_t to) {
  //std::cout << "TransposeSingleAxisOutwards" << std::endl;
  const auto& input_shape = input->getShape();
  //std::cout << "input_shape: " << input_shape.toString() << std::endl;
  const auto& input_dims = input_shape.getDims();
  //std::cout << "input_dims: " << MemoryDescUtils::dims2str(input_dims) << std::endl;
  const auto element_size = dnnl::memory::data_type_size(input->getDataType());
  //std::cout << "element_size: " << element_size << std::endl;

  const auto* input_data = reinterpret_cast<const uint8_t*>(input->getData());
  auto* output_data = reinterpret_cast<uint8_t*>(output->getData());

  auto num_loops = calcShapeSize(input_shape, 0, to);
  //std::cout << "num_loops: " << num_loops << std::endl;
  //std::cout << "from: " << from << std::endl;
  //std::cout << "to: " << to << std::endl;
  auto num_writers = input_dims[from];
  //std::cout << "num_writers: " << num_writers << std::endl;
  auto block_size = calcShapeSize(input_shape, from + 1, input_shape.getRank());
  //std::cout << "block_size: " << block_size << std::endl;
  auto writes_per_loop = int64_t(input_shape.getElementsCount() / num_loops / block_size);
  //std::cout << "input_shape.getElementsCount(): " << input_shape.getElementsCount() << std::endl;
  //std::cout << "writes_per_loop: " << writes_per_loop << std::endl;
  auto writes_per_writer_per_loop = int64_t(writes_per_loop / num_writers);
  //std::cout << "writes_per_writer_per_loop: " << writes_per_writer_per_loop << std::endl;
  // TODO: check integer overflow
  const size_t bytes_per_write = static_cast<size_t>(block_size) * element_size;

  switch (bytes_per_write) {
    case (sizeof(uint8_t)): {
      //std::cout << "uint8_t" << std::endl;
      SimpleTransposeSingleAxisOutwards(input_data, output_data, num_loops, num_writers, writes_per_loop,
                                        writes_per_writer_per_loop);
      break;
    }
    case (sizeof(uint16_t)): {
      //std::cout << "uint16_t" << std::endl;
      SimpleTransposeSingleAxisOutwards(reinterpret_cast<const uint16_t*>(input_data),
                                        reinterpret_cast<uint16_t*>(output_data), num_loops, num_writers,
                                        writes_per_loop, writes_per_writer_per_loop);
      break;
    }
    case (sizeof(uint32_t)): {
      //std::cout << "uint32_t" << std::endl;
      SimpleTransposeSingleAxisOutwards(reinterpret_cast<const uint32_t*>(input_data),
                                        reinterpret_cast<uint32_t*>(output_data), num_loops, num_writers,
                                        writes_per_loop, writes_per_writer_per_loop);
      break;
    }
    case (sizeof(uint64_t)): {
      //std::cout << "uint64_t" << std::endl;
      SimpleTransposeSingleAxisOutwards(reinterpret_cast<const uint64_t*>(input_data),
                                        reinterpret_cast<uint64_t*>(output_data), num_loops, num_writers,
                                        writes_per_loop, writes_per_writer_per_loop);
      break;
    }
    default: {
      //std::cout << "TransposeSingleAxisOutwards: default branch" << std::endl;
      // we need to use memcpy for each block
      for (int64_t l = 0; l < num_loops; ++l) {
        uint8_t* output_for_first_writer = output_data;

        for (auto wwpl = 0; wwpl < writes_per_writer_per_loop; ++wwpl) {
          uint8_t* output_for_current_writer = output_for_first_writer;

          for (int64_t w = 0; w < num_writers; ++w) {
            memcpy(output_for_current_writer, input_data, bytes_per_write);
            // skip to output position for next writer
            output_for_current_writer += (writes_per_writer_per_loop * bytes_per_write);
            input_data += bytes_per_write;
          }

          output_for_first_writer += bytes_per_write;
        }

        output_data += writes_per_loop * bytes_per_write;
      }
    }
  }
}

void RefTransposeExecutor::TransposeSingleAxisInwards(const MemoryCPtr& input, const MemoryPtr& output, size_t from, size_t to) {
  //std::cout << "TransposeSingleAxisInwards" << std::endl;
  const auto& input_shape = input->getShape();
  //std::cout << "input_shape: " << input_shape.toString() << std::endl;
  const auto& input_dims = input_shape.getDims();
  //std::cout << "input_dims: " << MemoryDescUtils::dims2str(input_dims) << std::endl;

  const auto element_size = dnnl::memory::data_type_size(input->getDataType());
  //std::cout << "element_size: " << element_size << std::endl;
  const auto* input_data = reinterpret_cast<const uint8_t*>(input->getData());
  auto* output_data = reinterpret_cast<uint8_t*>(output->getData());

  auto num_loops = calcShapeSize(input_shape, 0, from);
  //std::cout << "num_loops: " << num_loops << std::endl;
  //std::cout << "from: " << from << std::endl;
  //std::cout << "to: " << to << std::endl;
  auto num_readers = input_dims[from];
  //std::cout << "num_readers: " << num_readers << std::endl;
  auto block_size = calcShapeSize(input_shape, to + 1, input_shape.getRank());
  //std::cout << "block_size: " << block_size << std::endl;
  auto reads_per_loop = int64_t(input_shape.getElementsCount() / num_loops / block_size);
  //std::cout << "input_shape.getElementsCount(): " << input_shape.getElementsCount() << std::endl;
  //std::cout << "reads_per_loop: " << reads_per_loop << std::endl;
  auto reads_per_reader_per_loop = int64_t(reads_per_loop / num_readers);
  //std::cout << "reads_per_reader_per_loop: " << reads_per_reader_per_loop << std::endl;
  // TODO: check integer overflow
  const size_t bytes_per_read = static_cast<size_t>(block_size) * element_size;
  //std::cout << "bytes_per_read: " << bytes_per_read << std::endl;

  switch (bytes_per_read) {
    case (sizeof(uint8_t)): {
      SimpleTransposeSingleAxisInwards(input_data, output_data, num_loops, num_readers, reads_per_loop,
                                       reads_per_reader_per_loop);
      break;
    }
    case (sizeof(uint16_t)): {
      SimpleTransposeSingleAxisInwards(reinterpret_cast<const uint16_t*>(input_data),
                                       reinterpret_cast<uint16_t*>(output_data), num_loops, num_readers, reads_per_loop,
                                       reads_per_reader_per_loop);
      break;
    }
    case (sizeof(uint32_t)): {
      SimpleTransposeSingleAxisInwards(reinterpret_cast<const uint32_t*>(input_data),
                                       reinterpret_cast<uint32_t*>(output_data), num_loops, num_readers, reads_per_loop,
                                       reads_per_reader_per_loop);
      break;
    }
    case (sizeof(uint64_t)): {
      SimpleTransposeSingleAxisInwards(reinterpret_cast<const uint64_t*>(input_data),
                                       reinterpret_cast<uint64_t*>(output_data), num_loops, num_readers, reads_per_loop,
                                       reads_per_reader_per_loop);
      break;
    }
    default: {
      //std::cout << "TransposeSingleAxisInwards - default branch" << std::endl;
      // we need to use memcpy for each block
      for (int64_t l = 0; l < num_loops; ++l) {
        const uint8_t* input_for_first_reader = input_data;
        for (auto rrpl = 0; rrpl < reads_per_reader_per_loop; ++rrpl) {
          const uint8_t* input_for_current_reader = input_for_first_reader;
          for (int64_t r = 0; r < num_readers; ++r) {
            memcpy(output_data, input_for_current_reader, bytes_per_read);
            output_data += bytes_per_read;
            // skip to input position for next reader
            input_for_current_reader += (reads_per_reader_per_loop * bytes_per_read);
          }
          input_for_first_reader += bytes_per_read;
        }
        input_data += reads_per_loop * bytes_per_read;
      }
    }
  }
}

void RefTransposeExecutor::referenceExecute(const uint8_t* src_data, uint8_t* dst_data, jit_permute_config_params jcp, const int mb) {
    SizeVector dst_dims = jcp.dst_block_dims;
    const SizeVector dst_strides = jcp.dst_strides;
    const SizeVector src_strides = jcp.src_strides;
    const size_t data_size = jcp.data_size;
    const size_t ndims = dst_dims.size();

    if (static_cast<int>(dst_dims[0]) != mb)
        dst_dims[0] = mb;

    size_t work_amount = std::accumulate(dst_dims.begin(), dst_dims.end(), 1, std::multiplies<size_t>());

    auto get_idx = [ndims, data_size](const SizeVector& indexes, const SizeVector& strides) {
        size_t idx = 0;
        for (size_t i = 0; i < ndims; ++i)
            idx += indexes[i] * strides[i];
        return idx * data_size;
    };

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector indexes(ndims, 0);
        splitter(work_amount, nthr, ithr, start, end);

        parallel_init(start, ndims, dst_dims, indexes);

        for (size_t iwork = start; iwork < end; ++iwork) {
            const size_t dst_idx = get_idx(indexes, dst_strides);
            const size_t src_idx = get_idx(indexes, src_strides);
            cpu_memcpy(&dst_data[dst_idx], &src_data[src_idx], data_size);

            parallel_step(ndims, dst_dims, indexes);
        }
    });
}

void RefTransposeExecutor::SingleAxisTranspose(SizeVector permutations, const MemoryCPtr& input, const MemoryPtr& output, size_t from, size_t to) {
  //std::cout << "SingleAxisTranspose" << std::endl;
  if (from > to) {
    TransposeSingleAxisOutwards(input, output, from, to);
  } else {
    TransposeSingleAxisInwards(input, output, from, to);
  }
}

void RefTransposeExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) {
    size_t from = 0, to = 0;
    bool moving_single_axis = IsTransposeMovingSingleAxis(permutations, from, to);
    //std::cout << "moving_single_axis: " << moving_single_axis << std::endl;
    //TODO: confirm that MLAS supports NCHW only
    if (moving_single_axis && src[0]->getDesc().hasLayoutType(LayoutType::ncsp)) {
      SingleAxisTranspose(permutations, src[0], dst[0], from, to);
    } else {
      // fall back to default implementation
      //std::cout << "fall back to default implementation" << std::endl;
      const uint8_t* src_data = reinterpret_cast<const uint8_t*>(src[0]->getData());
      uint8_t* dst_data = reinterpret_cast<uint8_t*>(dst[0]->getData());
      referenceExecute(src_data, dst_data, jcp, MB);
    }
}

bool RefTransposeExecutor::init(const TransposeParams &transposeParams,
                                const std::vector<MemoryDescPtr> &srcDescs,
                                const std::vector<MemoryDescPtr> &dstDescs,
                                const dnnl::primitive_attr &attr) {
    jcp = TransposeExecutor::prepareParams(transposeParams.permuteParams);
    permutations = transposeParams.permuteParams.order;
    return true;
}

}   // namespace intel_cpu
}   // namespace ov