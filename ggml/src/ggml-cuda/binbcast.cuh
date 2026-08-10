#include "common.cuh"

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
void ggml_cuda_op_fused_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);

#define CUDA_BINBCAST_BATCH_MAX 16

// data pointers for a run of same-geometry independent binary ops
// (see ggml_cuda_mul_batched)
struct ggml_cuda_binbcast_batch_ptrs {
    const char * src0[CUDA_BINBCAST_BATCH_MAX];
    const char * src1[CUDA_BINBCAST_BATCH_MAX];
    char       * dst [CUDA_BINBCAST_BATCH_MAX];
};

// true when `node` is a MUL whose operands the batched route supports
bool ggml_cuda_mul_can_batch(const ggml_tensor * node);

// run n_batch same-geometry MULs in one launch; proto supplies the shared
// geometry (all run members were checked identical by the caller)
void ggml_cuda_mul_batched(ggml_backend_cuda_context & ctx, const ggml_tensor * proto,
        const ggml_cuda_binbcast_batch_ptrs & ptrs, int n_batch);
