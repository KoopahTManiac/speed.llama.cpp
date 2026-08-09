#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

// max copies per batched launch - bounded by the kernel-argument struct size
#define CUDA_CPY_BATCH_MAX 16

// data pointers for a run of same-geometry copies (see ggml_cuda_cpy_batched)
struct ggml_cuda_cpy_batch_ptrs {
    const char * src[CUDA_CPY_BATCH_MAX];
    char       * dst[CUDA_CPY_BATCH_MAX];
};

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

// one launch for a run of CPY nodes whose src/dst tensors all share the
// geometry (type, ne, nb) of src0_ref/src1_ref and differ only in their data
// pointers. Supported for same-type scalar copies; the caller checks
// eligibility with ggml_cuda_cpy_can_batch.
bool ggml_cuda_cpy_can_batch(const ggml_tensor * src0, const ggml_tensor * src1);

void ggml_cuda_cpy_batched(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0_ref, const ggml_tensor * src1_ref,
        const ggml_cuda_cpy_batch_ptrs & ptrs, int n_batch);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
