#include "dspark-sample.cuh"

// Fused truncation sampler (see ggml_dspark_sample): one block per column.
// The candidates are gathered and bitonic-sorted in shared memory; the
// O(n_cand) scalar tail (masks, softmaxes, inverse CDF) runs on a single
// thread - at n_cand <= 64 it is a few hundred flops and stays well below
// the cost of the ~25 separate kernel launches this op replaces.
// Tie ordering of equal logits follows the bitonic network (unstable), like
// the composed argsort-based chain on this backend.
static __global__ void dspark_sample_f32(
        const float * GGML_CUDA_RESTRICT logits, const int32_t * GGML_CUDA_RESTRICT cand,
        const float * GGML_CUDA_RESTRICT uniform, const float * GGML_CUDA_RESTRICT inv_temp,
        const float * GGML_CUDA_RESTRICT topk_mask, const float * GGML_CUDA_RESTRICT top_p, const float * GGML_CUDA_RESTRICT min_p,
        float * GGML_CUDA_RESTRICT dst,
        const int n_cand,
        const int64_t s_logits, const int64_t s_cand, const int64_t s_mask,
        const int64_t s_u, const int64_t s_it, const int64_t s_tp, const int64_t s_mp,
        const int64_t s_dst,
        const int emit_dist) {
    constexpr int W = GGML_CUDA_DSPARK_SAMPLE_MAX_CAND;

    const int j   = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float sv [W];
    __shared__ float sid[W];

    ggml_cuda_pdl_sync();

    // gather the candidate logits; pad the sort width with -inf
    if (tid < W) {
        if (tid < n_cand) {
            const int32_t id = cand[j*s_cand + tid];
            sv [tid] = logits[j*s_logits + id];
            sid[tid] = (float) id;
        } else {
            sv [tid] = -INFINITY;
            sid[tid] = 0.0f;
        }
    }
    __syncthreads();

    // bitonic sort, descending by value
    for (int k = 2; k <= W; k *= 2) {
        for (int s = k/2; s > 0; s /= 2) {
            const int ixj = tid ^ s;
            if (tid < W && ixj > tid) {
                const bool dir = (tid & k) == 0; // descending in the first half
                const bool swap = dir ? sv[tid] < sv[ixj] : sv[tid] > sv[ixj];
                if (swap) {
                    const float tv = sv[tid];  sv[tid]  = sv[ixj];  sv[ixj]  = tv;
                    const float ti = sid[tid]; sid[tid] = sid[ixj]; sid[ixj] = ti;
                }
            }
            __syncthreads();
        }
    }

    // additive top-k mask, indexed by sorted rank
    if (tid < n_cand) {
        sv[tid] += topk_mask[j*s_mask + tid];
    }
    __syncthreads();

    if (tid != 0) {
        return;
    }

    const float u_j  = uniform [j*s_u];
    const float it_j = inv_temp[j*s_it];
    const float tp_j = top_p   [j*s_tp];
    const float mp_j = min_p   [j*s_mp];

    // untempered softmax for the top-p / min-p keep decisions
    float vmax = -INFINITY;
    for (int c = 0; c < n_cand; ++c) vmax = fmaxf(vmax, sv[c]);
    float pf[W];
    float sum_f = 0.0f;
    for (int c = 0; c < n_cand; ++c) { pf[c] = expf(sv[c] - vmax); sum_f += pf[c]; }
    for (int c = 0; c < n_cand; ++c) { pf[c] /= sum_f; }

    // tempered softmax, masked by the keep decisions
    float tmax = -INFINITY;
    for (int c = 0; c < n_cand; ++c) tmax = fmaxf(tmax, sv[c]*it_j);
    float q[W];
    float sum_t = 0.0f;
    for (int c = 0; c < n_cand; ++c) { q[c] = expf(sv[c]*it_j - tmax); sum_t += q[c]; }

    float cum_f = 0.0f;
    float mass  = 0.0f;
    for (int c = 0; c < n_cand; ++c) {
        q[c] /= sum_t;
        const bool keep_p = -cum_f + tp_j > 0.0f; // cumulative mass before c
        cum_f += pf[c];
        const bool keep_m = pf[c] - pf[0]*mp_j > 0.0f;
        if (!(keep_p && keep_m)) {
            q[c] = 0.0f;
        }
        mass += q[c];
    }

    // inverse CDF at uniform * kept mass
    const float u = u_j * mass;
    float cum_q = 0.0f;
    int hits = 0;
    for (int c = 0; c < n_cand; ++c) {
        cum_q += q[c];
        hits += cum_q - u > 0.0f ? 1 : 0;
    }
    int pos = n_cand - hits;
    pos = min(max(pos, 0), n_cand - 1);

    float * out = dst + j*s_dst;
    out[0] = sid[pos];
    if (emit_dist) {
        const float inv_mass = mass > 0.0f ? 1.0f/mass : 0.0f;
        out[1] = q[pos]*inv_mass;
        for (int c = 0; c < n_cand; ++c) {
            out[2 + c]          = q[c]*inv_mass;
            out[2 + n_cand + c] = sid[c];
        }
    }
}

void ggml_cuda_op_dspark_sample(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * logits    = dst->src[0];
    const ggml_tensor * cand      = dst->src[1];
    const ggml_tensor * uniform   = dst->src[2];
    const ggml_tensor * inv_temp  = dst->src[3];
    const ggml_tensor * topk_mask = dst->src[4];
    const ggml_tensor * top_p     = dst->src[5];
    const ggml_tensor * min_p     = dst->src[6];

    const int64_t n_cols = logits->ne[1];
    const int64_t n_cand = cand->ne[0];
    GGML_ASSERT(n_cand <= GGML_CUDA_DSPARK_SAMPLE_MAX_CAND);

    const int emit_dist = ggml_get_op_params_i32(dst, 0);

    const dim3 grid((unsigned) n_cols, 1, 1);
    const dim3 block(GGML_CUDA_DSPARK_SAMPLE_MAX_CAND, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, 0, ctx.stream());

    ggml_cuda_kernel_launch(dspark_sample_f32, launch_params,
        (const float *) logits->data, (const int32_t *) cand->data,
        (const float *) uniform->data, (const float *) inv_temp->data,
        (const float *) topk_mask->data, (const float *) top_p->data, (const float *) min_p->data,
        (float *) dst->data,
        (int) n_cand,
        logits->nb[1]/sizeof(float), cand->nb[1]/sizeof(int32_t), topk_mask->nb[1]/sizeof(float),
        uniform->nb[1]/sizeof(float), inv_temp->nb[1]/sizeof(float),
        top_p->nb[1]/sizeof(float), min_p->nb[1]/sizeof(float),
        dst->nb[1]/sizeof(float),
        emit_dist);
}
