// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <map>

namespace ov {
namespace intel_cpu {
namespace node {

enum dft_type {
    real_to_complex,
    complex_to_complex,
    complex_to_real,
};

struct RDFTExecutor {
    public:
        RDFTExecutor(bool inverse) : is_inverse(inverse) {}
        void execute(float* src, float* dst, size_t rank,
                     const std::vector<int>& axes,
                     std::vector<int> signal_sizes,
                     VectorDims input_shape, const VectorDims& output_shape,
                     const VectorDims& input_strides, const VectorDims& output_strides);

    protected:
        std::vector<std::vector<float>> twiddles;
        bool is_inverse;

    private:
        virtual bool can_use_fft(size_t dim) = 0;
        virtual void dft(float* input_ptr, float* twiddles_ptr, float* output_ptr,
                         size_t input_size, size_t num_samples, size_t output_size,
                         enum dft_type type, bool parallelize) = 0;
        virtual void fft(float* input, float* twiddles_ptr, float* output,
                         size_t input_size, size_t num_samples, size_t output_size,
                         enum dft_type type, bool parallelize) = 0;
        void dft_common(float* input_ptr, float* twiddles_ptr, float* output_ptr,
                        size_t input_size, size_t num_samples, size_t output_size,
                        enum dft_type type, bool use_fft, bool parallelize);
        void dft_on_axis(enum dft_type type,
                         float* input_ptr, float* output_ptr,
                         float* twiddles_ptr, int axis,
                         size_t num_samples,
                         const VectorDims& input_shape,
                         const VectorDims& input_strides,
                         const VectorDims& output_shape,
                         const VectorDims& output_strides,
                         const std::vector<size_t>& iteration_range);
        void rdft_nd(float* input_ptr, float* output_ptr,
                     const std::vector<int>& axes,
                     const std::vector<int>& signal_sizes,
                     const VectorDims& input_shape,
                     const VectorDims& input_strides,
                     const VectorDims& output_shape,
                     const VectorDims& output_strides);
        void irdft_nd(float* input_ptr, float* output_ptr,
                      const std::vector<int>& axes,
                      const std::vector<int>& signal_sizes,
                      const VectorDims& input_shape,
                      const VectorDims& input_strides,
                      const VectorDims& output_shape,
                      const VectorDims& output_strides);
        virtual std::vector<float> generate_twiddles_dft(size_t input_size, size_t output_size, enum dft_type type) = 0;
        std::vector<float> generate_twiddles_fft(size_t N);
        std::vector<float> generate_twiddles_common(size_t input_size, size_t output_size,
                                                    enum dft_type type, bool use_fft);
        void generate_twiddles(const std::vector<int>& signal_sizes,
                               const std::vector<size_t>& output_shape,
                               const std::vector<int>& axes);
};

class RDFT : public Node {
public:
    RDFT(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool needPrepareParams() const override;
    void prepareParams() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    std::string error_msg_prefix;

    bool inverse;

    std::shared_ptr<RDFTExecutor> executor;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
