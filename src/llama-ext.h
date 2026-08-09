#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>
#include <vector>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

//
// DSpark draft sampling
//

// Row layout of the DSpark draft's nextn channel (see build_dspark_markov_head):
// row 0 holds each drafted position's acceptance confidence, row 1 the chain's
// chosen token id (float-encoded; vocab sizes fit exactly in f32's integer
// range), row 2 the proposal probability of the chosen token, and - when the
// channel is wide enough (n_embd >= 3 + 2*n_cand) - the proposal distribution
// itself: n_cand normalized candidate probabilities followed by n_cand
// candidate ids. The distribution feeds exact ratio acceptance.
#define LLAMA_DSPARK_NEXTN_ROW_CONF  0
#define LLAMA_DSPARK_NEXTN_ROW_TOKEN 1
#define LLAMA_DSPARK_NEXTN_ROW_Q     2
#define LLAMA_DSPARK_NEXTN_ROW_CAND  3

// Per-sequence sampling configuration for the DSpark draft's markov chain,
// mirroring the request's sampling parameters. temp <= 0 drafts greedily
// (argmax); temp > 0 samples each position by inverse-CDF over the sorted
// candidate set at the given temperature (the same technique as the dist
// backend sampler), truncated by top_k (0 = disabled) and top_p (1.0 =
// disabled), each position conditioned on the token actually sampled at the
// previous one. All values are graph inputs: changing them does not rebuild
// the graph.
// Note: proposals are always truncated to the context's candidate capacity
// (llama_set_dspark_draft_n_cand); a top_k of 0 or beyond the capacity
// proposes from the top-capacity set. This only affects acceptance rates,
// never output correctness.
struct llama_dspark_draft_sampling {
    float    temp;
    float    top_p;
    float    min_p;
    int32_t  top_k;
    uint32_t seed;
};

// The request's seed drives several uniform streams: the draft chain's picks,
// the target's verify picks and residual draws, and the caller's acceptance
// tests. Exact ratio acceptance requires them to be mutually independent, so
// each role seeds its generator through std::seed_seq{seed, role} instead of
// the raw seed (identical raw-seeded streams would correlate the acceptance
// draw with the proposal draw and bias the emitted distribution).
enum llama_dspark_rng_role : uint32_t {
    LLAMA_DSPARK_RNG_ROLE_CHAIN  = 0, // draft chain sampling
    LLAMA_DSPARK_RNG_ROLE_VERIFY = 1, // target verify sampling + residual
    LLAMA_DSPARK_RNG_ROLE_ACCEPT = 2, // caller's ratio-acceptance tests
};

// Configure the drafting distribution for one sequence of a DSpark draft
// context. Takes effect on the next decode.
LLAMA_API void llama_set_dspark_draft_sampling(
        struct llama_context * ctx,
        llama_seq_id           seq_id,
        struct llama_dspark_draft_sampling sampling);

// Set the candidate-set capacity of the sampled chain. The speculative driver
// derives this from the largest top-k across the sequences it serves (falling
// back to its configured sampling defaults for sequences without top-k).
// Structural: changing it rebuilds the draft graph; set before the first
// decode. 0 (default) disables the sampled chain: drafting is greedy.
LLAMA_API void llama_set_dspark_draft_n_cand(struct llama_context * ctx, uint32_t n_cand);

LLAMA_API uint32_t llama_get_dspark_draft_n_cand(const struct llama_context * ctx);

// Make decode-path embd batches carry encoder-width rows (raw target
// features): the decoder graph applies the feature-fusion projection
// in-graph, so drivers submit one llama_decode instead of llama_encode +
// readback + llama_decode. Returns false when the model's decoder does not
// support the fused path (callers must then keep the two-call flow).
LLAMA_API bool llama_set_decode_embd_enc(struct llama_context * ctx, bool value);

// Speculative verify sampling: sample every output row of a decode in-graph
// with the row's per-sequence config (same configs and capacity as above, set
// on the TARGET context). The sampled ids are read with
// llama_get_spec_verify_sampled_ith, indexed like llama_get_logits_ith.
// Requests whose sampler chains go beyond temp/top-k/top-p/min-p (grammar,
// penalties, ...) must use the host sampling path instead.
LLAMA_API void llama_set_spec_verify_sampling(struct llama_context * ctx, bool value);

// Returns the in-graph sampled token for output row i of the last decode, or
// -1 when verify sampling was not active for that decode.
LLAMA_API llama_token llama_get_spec_verify_sampled_ith(struct llama_context * ctx, int32_t i);

// Declare that a sequence consumes its verify decodes exclusively through the
// getters above: when every output row of a decode belongs to an opted-out
// sequence (or one with a backend sampler), the full-vocabulary logits copy
// to the host is skipped. Callers must clear the flag before decodes whose
// logits they read on the host again (prompt processing, host-sampled rows).
LLAMA_API void llama_set_spec_verify_backend_only(struct llama_context * ctx, llama_seq_id seq_id, bool value);

// A draft's proposal distribution for one sequence's next verify decode:
// n_pos positions, each with n_cand normalized candidate probabilities
// followed by n_cand candidate ids (float-encoded), laid out position-major.
struct llama_spec_verify_draft_dist {
    uint32_t n_pos  = 0;
    uint32_t n_cand = 0;
    std::vector<float> data;
};

// Provide the draft's proposal distribution for the next verify decode of a
// sequence. Enables exact ratio acceptance: the verify graph computes the
// served probability of each draft token and an exact residual sample.
LLAMA_API void llama_set_spec_verify_draft_dist(
        struct llama_context * ctx,
        llama_seq_id           seq_id,
        const float          * data,
        uint32_t               n_pos,
        uint32_t               n_cand);

// Returns ratio-acceptance outputs for output row i of the last decode:
// *p_draft = the served probability of the draft token evaluated at that row,
// *residual = the exact residual sample to emit when that row rejects.
// Returns false when ratio outputs were not produced for that decode.
LLAMA_API bool llama_get_spec_verify_ratio_ith(
        struct llama_context * ctx,
        int32_t                i,
        float                * p_draft,
        llama_token          * residual);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
