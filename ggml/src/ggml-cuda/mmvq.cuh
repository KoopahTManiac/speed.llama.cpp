#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which the MMVQ MoE kernel should be used
// for MUL_MAT_ID. Types with a tuned crossover below MMVQ_MAX_BATCH_SIZE keep
// it; all other types extend to the kernel's launch-geometry capability
// (device max_threads_per_block / warp_size), because the only alternative for
// larger batches is the generic fallback, which synchronizes the stream per
// call and prevents CUDA graph capture.
int get_mmvq_mmid_max_batch(ggml_type type, const ggml_cuda_device_info::cuda_device_info & device);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
