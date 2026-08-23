// Copyright © 2024 Apple Inc.

#include <metal_simdgroup>

using namespace metal;

constant bool has_mask [[function_constant(20)]];
constant bool query_transposed [[function_constant(21)]];
constant bool do_causal [[function_constant(22)]];
constant bool bool_mask [[function_constant(23)]];
constant bool float_mask [[function_constant(24)]];
constant bool has_sinks [[function_constant(25)]];
constant int blocks [[function_constant(26)]];

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device bool* bmask [[buffer(11), function_constant(bool_mask)]],
    const device T* fmask [[buffer(12), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(13), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(14), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(15), function_constant(has_mask)]],
    const device T* sinks [[buffer(16), function_constant(has_sinks)]],
    const constant int& num_q_heads
    [[buffer(17), function_constant(has_sinks)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device bool* bmask [[buffer(13), function_constant(bool_mask)]],
    const device T* fmask [[buffer(14), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(15), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(16), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(17), function_constant(has_mask)]],
    const device T* sinks [[buffer(18), function_constant(has_sinks)]],
    // Full query sequence length. The threadgroup z extent covers only a
    // CHUNK of the query positions when q_seq_len * gqa_factor would exceed
    // the 1024-thread cap; tid.y then packs (batch, q-chunk). Causal
    // alignment and the output stride must use the full length, not tptg.z.
    const constant int& q_seq_len_param [[buffer(19)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = q_seq_len_param;
  // Unpack (batch, q-chunk) from tid.y; this threadgroup's z covers
  // [q_chunk_idx * tptg.z, ...) of the full query sequence.
  const int num_q_chunks = (q_seq_len + (int)tptg.z - 1) / (int)tptg.z;
  const int batch_idx = tid.y / num_q_chunks;
  const int q_chunk_idx = tid.y % num_q_chunks;
  const int q_seq_idx = q_chunk_idx * (int)tptg.z + (int)tidtg.z;
  if (q_seq_idx >= q_seq_len) {
    // Tail chunk shorter than the threadgroup z extent: no barriers in this
    // kernel, so an early return is safe (no loads/stores may happen below).
    return;
  }
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset =
      query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = block_idx; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Compute the i-th score
      U score = 0;
      for (int i = 0; i < qk_per_thread; i++) {
        score += q[i] * keys[i];
      }
      score = simd_sum(score);

      if (float_mask) {
        score += fmask[0];
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int i = 0; i < v_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * values[i];
      }
    }

    // Move the pointers to the next kv
    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
    if (bool_mask) {
      bmask += blocks * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += blocks * mask_kv_seq_stride;
    }
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

// Multi-query variant of sdpa_vector_2pass_1 for short query blocks
// (speculative verify): each simdgroup holds QPS consecutive query positions
// in registers and streams every K/V row of its block ONCE, amortizing the
// dominant cost — the per-simdgroup KV stream — across those queries (the
// single-query kernel re-streams the block's K/V once per query, so its cost
// grows linearly in qL). Per-query arithmetic (element partitioning,
// simd_sum reduction, softmax accumulation order over keys) is identical to
// sdpa_vector_2pass_1, so each query's partials are the same values the
// single-query kernel produces.
template <typename T, int D, int V = D, int QPS = 2>
[[kernel]] void sdpa_vector_2pass_1_mq(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device bool* bmask [[buffer(13), function_constant(bool_mask)]],
    const device T* fmask [[buffer(14), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(15), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(16), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(17), function_constant(has_mask)]],
    const device T* sinks [[buffer(18), function_constant(has_sinks)]],
    const constant int& q_seq_len_param [[buffer(19)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  thread U q[QPS][qk_per_thread];
  thread U o[QPS][v_per_thread];
  thread U max_score[QPS];
  thread U sum_exp_score[QPS];

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = q_seq_len_param;
  // Each z slice covers QPS consecutive query positions; tid.y packs
  // (batch, q-chunk) exactly like the single-query kernel.
  const int q_per_tg = (int)tptg.z * QPS;
  const int num_q_chunks = (q_seq_len + q_per_tg - 1) / q_per_tg;
  const int batch_idx = tid.y / num_q_chunks;
  const int q_chunk_idx = tid.y % num_q_chunks;
  const int q_group_base = q_chunk_idx * q_per_tg + (int)tidtg.z * QPS;
  if (q_group_base >= q_seq_len) {
    // No barriers in this kernel, so an early return is safe.
    return;
  }
  const int nq = min(QPS, q_seq_len - q_group_base);
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_row = q_batch_head_idx * q_seq_len + q_group_base;
  const int q_row =
      query_transposed ? num_q_heads * q_group_base + q_batch_head_idx : o_row;
  const int q_row_stride = query_transposed ? num_q_heads : 1;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_group_base * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_group_base * mask_q_seq_stride;
  }

  // Read the group's queries and init the accumulators
  for (int j = 0; j < QPS; j++) {
    max_score[j] = Limits<U>::finite_min;
    sum_exp_score[j] = 0;
    for (int i = 0; i < v_per_thread; i++) {
      o[j][i] = 0;
    }
    if (j < nq) {
      const device T* qj =
          queries + (q_row + j * q_row_stride) * D + simd_lid * qk_per_thread;
      for (int i = 0; i < qk_per_thread; i++) {
        q[j][i] = static_cast<U>(scale) * qj[i];
      }
    } else {
      for (int i = 0; i < qk_per_thread; i++) {
        q[j][i] = 0;
      }
    }
  }
  if (has_sinks && block_idx == 0) {
    for (int j = 0; j < nq; j++) {
      max_score[j] = static_cast<U>(sinks[q_head_idx]);
      sum_exp_score[j] = 1;
    }
  }

  // For each key: one load, applied to every query in the group
  for (int i = block_idx; i < N; i += blocks) {
    bool use_key_j[QPS];
    bool any_key = false;
    for (int j = 0; j < QPS; j++) {
      bool use_key = j < nq;
      if (use_key) {
        if (do_causal) {
          use_key = i <= (N - q_seq_len + q_group_base + j);
        } else if (bool_mask) {
          use_key = bmask[j * mask_q_seq_stride];
        } else if (float_mask) {
          use_key =
              (fmask[j * mask_q_seq_stride] >= Limits<T>::finite_min);
        }
      }
      use_key_j[j] = use_key;
      any_key = any_key || use_key;
    }

    if (any_key) {
      U kvals[qk_per_thread];
      U vvals[v_per_thread];
      for (int e = 0; e < qk_per_thread; e++) {
        kvals[e] = keys[e];
      }
      for (int e = 0; e < v_per_thread; e++) {
        vvals[e] = values[e];
      }
      for (int j = 0; j < QPS; j++) {
        if (!use_key_j[j]) {
          continue;
        }
        // Compute the i-th score for query j
        U score = 0;
        for (int e = 0; e < qk_per_thread; e++) {
          score += q[j][e] * kvals[e];
        }
        score = simd_sum(score);

        if (float_mask) {
          score += fmask[j * mask_q_seq_stride];
        }

        // Update the accumulators
        U new_max = max(max_score[j], score);
        U factor = fast::exp(max_score[j] - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score[j] = new_max;
        sum_exp_score[j] = sum_exp_score[j] * factor + exp_score;

        for (int e = 0; e < v_per_thread; e++) {
          o[j][e] = o[j][e] * factor + exp_score * vvals[e];
        }
      }
    }

    // Move the pointers to the next kv
    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
    if (bool_mask) {
      bmask += blocks * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += blocks * mask_kv_seq_stride;
    }
  }

  // Write the sums, maxes and outputs
  for (int j = 0; j < nq; j++) {
    if (simd_lid == 0) {
      sums[(o_row + j) * blocks + block_idx] = sum_exp_score[j];
      maxs[(o_row + j) * blocks + block_idx] = max_score[j];
    }
    device T* oj =
        out + (o_row + j) * blocks * V + block_idx * V + simd_lid * v_per_thread;
    for (int e = 0; e < v_per_thread; e++) {
      oj[e] = static_cast<T>(o[j][e]);
    }
  }
}

// GQA-packed MMA pass 1 for the speculative-verify shape (qL == 8, big
// head_dim): ONE threadgroup carries every query row that shares a KV head
// (gqa q-heads x 8 qL = up to 8x8-row MMA stripes), so the KV stream is
// read once and every dot product runs on simdgroup MMA instead of the
// vector kernels' per-lane FMA chains (issue-bound at D == 256). Layout:
// tidtg.z picks the q-head stripe (8 rows), tidtg.y picks the D-dhalf owned
// for the output accumulation; a stripe's two halves each compute the
// partial scores over their 128 dims and exchange them through
// threadgroup memory, so S = Q.K^T is summed exactly once. Keys advance in
// contiguous 32-key blocks per partition (`blocks` partitions over the key
// axis, function constant 26); the merge in sdpa_vector_2pass_2 is
// order-agnostic, so contiguous partitions coexist with the strided
// vector kernels. Rows do online softmax redundantly per dhalf (identical
// inputs, no exchange); the O rescale factor crosses lanes through the
// per-stripe factor slots because a lane's softmax row (lane / 4) is not
// its C-fragment row. K/V fragments load straight from the KV cache
// (simdgroup_load, transpose for K^T) after a coalesced threadgroup
// stage per block, zero-padded past N (the cache allocation can end flush
// at N). Causal only, no mask arrays, no sinks,
// qL == 8 exactly (fragment rows are not clamped) — the host routes
// everything else to the vector kernels. fp32 accumulation throughout.
template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_1_mma(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device bool* bmask [[buffer(13), function_constant(bool_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(15), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(16), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(17), function_constant(has_mask)]],
    const constant int& q_seq_len_param [[buffer(19)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK = 32; // keys per block
  constexpr int QL = 8; // query rows per stripe (enforced by the host gate)
  constexpr int H2 = D / 2; // dims owned per dhalf
  constexpr int NT = H2 / 8; // 8x8 O tiles per dhalf

  const int gqa = (int)tptg.z;
  const int stripe = (int)tidtg.z; // q head within the KV-head group
  const int dhalf = (int)tidtg.y; // D-dhalf this simdgroup accumulates
  const int lane = (int)simd_lid;

  const int kv_head_idx = (int)tid.x;
  const int batch_idx = (int)tid.y;
  const int part_idx = (int)tid.z;

  // sS: per-dhalf partial scores (added after the barrier); sP: the
  // softmaxed probabilities re-staged for the PV MMAs; sFactor: per-row O
  // rescale factors; sKV: the current key block, then the current value
  // block — staged coalesced and re-read as fragments, zero-padded past N.
  // Extents cover gqa <= 6 at BK == 32.
  threadgroup float sS[2 * 6 * QL * BK];
  threadgroup T sP[6 * QL * BK];
  threadgroup float sFactor[6 * QL];
  // uint4-backed so the staging runs 16-byte loads (2-byte loads leave the
  // stream load-issue-bound at a quarter of the machine's rate).
  threadgroup uint4 sKV4[BK * D * 2 / 16];
  threadgroup T* sKV = (threadgroup T*)sKV4;

  const int num_kv_heads = (int)tpg.x;
  const int num_q_heads = num_kv_heads * gqa;
  const int q_head_idx = gqa * kv_head_idx + stripe;
  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
  const int q_seq_len = q_seq_len_param;

  // Contiguous partition of the key axis, extended to whole blocks; keys
  // read past pEnd belong to the next partition and are masked, not
  // skipped, so every partition sums an exact, disjoint key range.
  const int span_blocks = (N + blocks * BK - 1) / (blocks * BK);
  const int span = span_blocks * BK;
  const int p0 = part_idx * span;
  const int pEnd = min(p0 + span, N);

  const device T* kHead = keys + (size_t)(batch_idx * num_kv_heads + kv_head_idx) * k_head_stride;
  const device T* vHead = values + (size_t)(batch_idx * num_kv_heads + kv_head_idx) * v_head_stride;

  // Query fragments for this stripe's rows over this dhalf's dims, resident
  // for the whole key loop. Row stride handles both query layouts.
  const int q_ld = query_transposed ? num_q_heads * D : D;
  const device T* qBase = queries +
      (query_transposed
           ? (size_t)(batch_idx * num_q_heads + q_head_idx) * D
           : (size_t)q_batch_head_idx * 8 * D); // host pads q to 8 rows
  simdgroup_matrix<T, 8, 8> Qf[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    simdgroup_load(
        Qf[t], qBase, q_ld, ulong2((ulong)(dhalf * H2 + t * 8), 0));
  }

  // The C-fragment row this lane's thread_elements live on (Metal 8x8
  // simdgroup_matrix layout; same mapping as affine_qmm_mma8) — NOT the
  // row this lane owns in the scalar softmax below.
  const int fragRow = (int)(((lane >> 2) & 4) + ((lane >> 1) & 3));
  // Scalar softmax ownership: lane r*4+g owns row r's g-th quarter of
  // the block's columns.
  const int smRow = lane >> 2;
  const int smCol = (lane & 3) * (BK / 4);

  simdgroup_matrix<float, 8, 8> O[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    O[t] = simdgroup_matrix<float, 8, 8>(0);
  }
  float mRun = Limits<float>::finite_min; // running max of row smRow
  float lRun = 0; // running exp-sum of row smRow
  const int rowPos = N - q_seq_len + smRow; // causal limit of row smRow
  // Bool-mask row for smRow (padded rows clamp to the last real row —
  // their outputs are discarded by the guarded writeout).
  const device bool* bmaskRow = bool_mask
      ? bmask + q_batch_head_idx * mask_head_stride +
          min(smRow, q_seq_len - 1) * mask_q_seq_stride
      : nullptr;

  threadgroup float* sSMine = sS + (dhalf * gqa + stripe) * (QL * BK);
  threadgroup float* sSOther = sS + ((1 - dhalf) * gqa + stripe) * (QL * BK);
  threadgroup T* sPMine = sP + stripe * (QL * BK);

  const int tix = (int)(tidtg.z * 64 + tidtg.y * 32 + lane);
  const int nthreads = (int)(tptg.y * tptg.z) * 32;

  for (int n0 = p0; n0 < pEnd; n0 += BK) {
    // Stage the key block coalesced in 16-byte lines (transposed fragment
    // gathers straight from device memory are what killed the direct
    // variant), zero-padded past N so the ragged block needs no special
    // path. A line never crosses a key row (D * 2 is a multiple of 16).
    constexpr int V4R = D * 2 / 16; // uint4 lines per key row
    for (int i = tix; i < BK * V4R; i += nthreads) {
      const int n = i / V4R;
      const int c4 = i % V4R;
      uint4 val = uint4(0);
      if (n0 + n < N) {
        val = *((const device uint4*)(kHead + (size_t)(n0 + n) * k_seq_stride) + c4);
      }
      sKV4[i] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Partial scores over this dhalf's dims: S_half[8, BK].
    simdgroup_matrix<float, 8, 8> Sc[BK / 8];
#pragma unroll
    for (int c = 0; c < BK / 8; ++c) {
      Sc[c] = simdgroup_matrix<float, 8, 8>(0);
    }
#pragma unroll
    for (int c = 0; c < BK / 8; ++c) {
      for (int t = 0; t < NT; ++t) {
        simdgroup_matrix<T, 8, 8> Kf;
        simdgroup_load(
            Kf, sKV, (ulong)D,
            ulong2((ulong)(dhalf * H2 + t * 8), (ulong)(c * 8)), true);
        simdgroup_multiply_accumulate(Sc[c], Qf[t], Kf, Sc[c]);
      }
    }
#pragma unroll
    for (int c = 0; c < BK / 8; ++c) {
      simdgroup_store(Sc[c], sSMine, (ulong)BK, ulong2((ulong)(c * 8), 0));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax on row smRow (both halves redundantly — identical
    // inputs, saves an exchange). Masked: beyond this partition, beyond
    // the causal limit, or past N.
    float sv[BK / 4];
    float rowMax = Limits<float>::finite_min;
#pragma unroll
    for (int j = 0; j < BK / 4; ++j) {
      const int c = smCol + j;
      const int kpos = n0 + c;
      float s = (sSMine[smRow * BK + c] + sSOther[smRow * BK + c]) * scale;
      const bool masked = kpos >= pEnd || kpos >= N ||
          (do_causal && kpos > rowPos) ||
          (bool_mask && !bmaskRow[kpos * mask_kv_seq_stride]);
      sv[j] = masked ? Limits<float>::finite_min : s;
      rowMax = max(rowMax, sv[j]);
    }
    rowMax = max(rowMax, simd_shuffle_xor(rowMax, 1));
    rowMax = max(rowMax, simd_shuffle_xor(rowMax, 2));
    const float mNew = max(mRun, rowMax);
    // finite_min stays finite_min: fully-masked blocks contribute factor 1
    // and zero probabilities rather than NaNs.
    const float factor =
        mRun == Limits<float>::finite_min ? 1.0f : fast::exp(mRun - mNew);
    float rowSum = 0;
#pragma unroll
    for (int j = 0; j < BK / 4; ++j) {
      const float p = sv[j] == Limits<float>::finite_min
          ? 0.0f
          : fast::exp(sv[j] - mNew);
      sv[j] = p;
      rowSum += p;
    }
    rowSum += simd_shuffle_xor(rowSum, 1);
    rowSum += simd_shuffle_xor(rowSum, 2);
    lRun = lRun * factor + rowSum;
    mRun = mNew;
    if (dhalf == 0) {
      if ((lane & 3) == 0) {
        sFactor[stripe * QL + smRow] = factor;
      }
#pragma unroll
      for (int j = 0; j < BK / 4; ++j) {
        sPMine[smRow * BK + smCol + j] = (T)sv[j];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage the value block over the consumed key staging, same 16-byte
    // lines.
    for (int i = tix; i < BK * V4R; i += nthreads) {
      const int n = i / V4R;
      const int c4 = i % V4R;
      uint4 val = uint4(0);
      if (n0 + n < N) {
        val = *((const device uint4*)(vHead + (size_t)(n0 + n) * v_seq_stride) + c4);
      }
      sKV4[i] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Rescale the O accumulators by this block's factor (per C-fragment
    // row, crossed over via sFactor), then accumulate P.V.
    const float oFactor = sFactor[stripe * QL + fragRow];
    simdgroup_matrix<T, 8, 8> Pf[BK / 8];
#pragma unroll
    for (int c = 0; c < BK / 8; ++c) {
      simdgroup_load(Pf[c], sPMine, (ulong)BK, ulong2((ulong)(c * 8), 0));
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      O[t].thread_elements()[0] *= oFactor;
      O[t].thread_elements()[1] *= oFactor;
      for (int c = 0; c < BK / 8; ++c) {
        simdgroup_matrix<T, 8, 8> Vf;
        simdgroup_load(
            Vf, sKV, (ulong)D,
            ulong2((ulong)(dhalf * H2 + t * 8), (ulong)(c * 8)));
        simdgroup_multiply_accumulate(O[t], Pf[c], Vf, O[t]);
      }
    }
    // sS/sP/sKV are rewritten next block behind this barrier; the factor
    // slot is consumed above.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Unnormalized partials + per-row exp-sums and maxes, exactly the
  // contract sdpa_vector_2pass_2 merges.
  const int row0 = q_batch_head_idx * q_seq_len;
  device T* pOut = out + ((size_t)row0 * blocks + part_idx) * D;
  // Rows beyond the real q_seq_len are padding (host pads q to 8 rows for
  // qL < 8): stage each O tile per-simdgroup and copy only real rows out.
  // At qL = 8 every row copies — bitwise the simdgroup_store this replaces.
  threadgroup T* oStage = (threadgroup T*)sSMine; // free after the last barrier
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    simdgroup_matrix<T, 8, 8> Ot;
    Ot.thread_elements()[0] = (T)O[t].thread_elements()[0];
    Ot.thread_elements()[1] = (T)O[t].thread_elements()[1];
    simdgroup_store(Ot, oStage, (ulong)8, ulong2(0, 0));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    {
      const int r = (int)lane >> 2;      // 8 rows x 4 lanes: 2 cols per lane
      const int c = ((int)lane & 3) * 2;
      if (r < q_seq_len) {
        pOut[(size_t)r * blocks * D + dhalf * H2 + t * 8 + c] =
            oStage[r * 8 + c];
        pOut[(size_t)r * blocks * D + dhalf * H2 + t * 8 + c + 1] =
            oStage[r * 8 + c + 1];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (dhalf == 0 && (lane & 3) == 0 && smRow < q_seq_len) {
    sums[(row0 + smRow) * blocks + part_idx] = lRun;
    maxs[(row0 + smRow) * blocks + part_idx] = mRun;
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& blocks [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  // Set defaults
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  // Reduce the max
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  // Reduce the d
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  // Reduce the sum exp and partials
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  // Use shared memory to transpose and reduce the final block
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
