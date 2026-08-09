#include "common.cuh"

// max candidate count the fused sampler supports (shared-memory sort width)
#define GGML_CUDA_DSPARK_SAMPLE_MAX_CAND 64

void ggml_cuda_op_dspark_sample(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
