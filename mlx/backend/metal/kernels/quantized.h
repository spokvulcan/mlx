// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline void
qouter(const thread uint8_t* w, U x, U scale, U bias, thread U* result) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  if (bits == 2) {
    U s[4] = {scale, scale / 4.0f, scale / 16.0f, scale / 64.0f};
    for (int i = 0; i < (values_per_thread / 4); i++) {
      result[4 * i] += x * (s[0] * (w[i] & 0x03) + bias);
      result[4 * i + 1] += x * (s[1] * (w[i] & 0x0c) + bias);
      result[4 * i + 2] += x * (s[2] * (w[i] & 0x30) + bias);
      result[4 * i + 3] += x * (s[3] * (w[i] & 0xc0) + bias);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[8 * i] += x * ((w0 & 0x7) * scale + bias);
      result[8 * i + 1] += x * (((w0 & 0x38) >> 3) * scale + bias);
      result[8 * i + 2] +=
          x * ((((w0 & 0xc0) >> 6) + ((w1 & 0x1) << 2)) * scale + bias);
      result[8 * i + 3] += x * (((w1 & 0xe) >> 1) * scale + bias);
      result[8 * i + 4] += x * (((w1 & 0x70) >> 4) * scale + bias);
      result[8 * i + 5] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0x3) << 1)) * scale + bias);
      result[8 * i + 6] += x * (((w2 & 0x1c) >> 2) * scale + bias);
      result[8 * i + 7] += x * (((w2 & 0xe0) >> 5) * scale + bias);
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / 16.0f};
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += x * (s[0] * (w[i] & 0x0f) + bias);
      result[2 * i + 1] += x * (s[1] * (w[i] & 0xf0) + bias);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[5 * i];
      uint8_t w1 = w[5 * i + 1];
      uint8_t w2 = w[5 * i + 2];
      uint8_t w3 = w[5 * i + 3];
      uint8_t w4 = w[5 * i + 4];
      result[8 * i] += x * ((w0 & 0x1f) * scale + bias);
      result[8 * i + 1] +=
          x * ((((w0 & 0xe0) >> 5) + ((w1 & 0x3) << 3)) * scale + bias);
      result[8 * i + 2] += x * (((w1 & 0x7c) >> 2) * scale + bias);
      result[8 * i + 3] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0xf) << 1)) * scale + bias);
      result[8 * i + 4] +=
          x * ((((w2 & 0xf0) >> 4) + ((w3 & 0x1) << 4)) * scale + bias);
      result[8 * i + 5] += x * (((w3 & 0x3e) >> 1) * scale + bias);
      result[8 * i + 6] +=
          x * ((((w3 & 0xc0) >> 6) + ((w4 & 0x7) << 2)) * scale + bias);
      result[8 * i + 7] += x * (((w4 & 0xf8) >> 3) * scale + bias);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[4 * i] += x * ((w0 & 0x3f) * scale + bias);
      result[4 * i + 1] +=
          x * ((((w0 >> 6) & 0x03) + ((w1 & 0x0f) << 2)) * scale + bias);
      result[4 * i + 2] +=
          x * ((((w1 >> 4) & 0x0f) + ((w2 & 0x03) << 4)) * scale + bias);
      result[4 * i + 3] += x * (((w2 >> 2) & 0x3f) * scale + bias);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
}

// Decode one quantized block (scale * q + bias) into w_local. W (the output
// pointer type) serves the threadgroup block loader or a thread-local decode.
template <typename U, int N, int bits, typename W>
inline void dequantize(const device uint8_t* w, U scale, U bias, W w_local) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  if (bits == 2) {
    U s[4] = {
        scale,
        scale / static_cast<U>(4.0f),
        scale / static_cast<U>(16.0f),
        scale / static_cast<U>(64.0f)};
    for (int i = 0; i < (N / 4); i++) {
      w_local[4 * i] = s[0] * (w[i] & 0x03) + bias;
      w_local[4 * i + 1] = s[1] * (w[i] & 0x0c) + bias;
      w_local[4 * i + 2] = s[2] * (w[i] & 0x30) + bias;
      w_local[4 * i + 3] = s[3] * (w[i] & 0xc0) + bias;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 3 * i;

      w_local[0] = (w[0] & 0x7) * scale + bias;
      w_local[1] = ((w[0] & 0x38) >> 3) * scale + bias;
      w_local[2] = (((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * scale + bias;
      w_local[3] = ((w[1] & 0xe) >> 1) * scale + bias;
      w_local[4] = ((w[1] & 0x70) >> 4) * scale + bias;
      w_local[5] = (((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * scale + bias;
      w_local[6] = ((w[2] & 0x1c) >> 2) * scale + bias;
      w_local[7] = ((w[2] & 0xe0) >> 5) * scale + bias;
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / static_cast<U>(16.0f)};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = s[0] * (w[i] & 0x0f) + bias;
      w_local[2 * i + 1] = s[1] * (w[i] & 0xf0) + bias;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;

      w_local[0] = (w[0] & 0x1f) * scale + bias;
      w_local[1] = (((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * scale + bias;
      w_local[2] = ((w[1] & 0x7c) >> 2) * scale + bias;
      w_local[3] = (((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * scale + bias;
      w_local[4] = (((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * scale + bias;
      w_local[5] = ((w[3] & 0x3e) >> 1) * scale + bias;
      w_local[6] = (((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * scale + bias;
      w_local[7] = ((w[4] & 0xf8) >> 3) * scale + bias;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = (w[0] & 0x3f) * scale + bias;
      w_local[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
      w_local[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
      w_local[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = scale * w[i] + bias;
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoader {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

template <typename T, int group_size, int bits, int D>
METAL_FUNC void qmv_quad_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  constexpr int quads_per_simd = SIMD_SIZE / QUAD_SIZE;
  constexpr int pack_factor = 32 / bits;
  constexpr int values_per_thread = D / QUAD_SIZE;
  constexpr int packs_per_thread = values_per_thread / pack_factor;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int results_per_quadgroup = 8;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_quadgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * quads_per_simd * results_per_quadgroup + quad_gid;

  w += out_row * in_vec_size_w + quad_lid * packs_per_thread;
  scales += out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + quad_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

  for (int row = 0; row < results_per_quadgroup; row++) {
    auto wl = (const device uint8_t*)(w + row * in_vec_size_w * quads_per_simd);
    const device T* sl = scales + row * in_vec_size_g * quads_per_simd;
    const device T* bl = biases + row * in_vec_size_g * quads_per_simd;

    U s = sl[0];
    U b = bl[0];
    if (row * quads_per_simd + out_row < out_vec_size) {
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
  }

  for (int row = 0; row < results_per_quadgroup; row++) {
    result[row] = quad_sum(result[row]);
    if (quad_lid == 0 && row * quads_per_simd + out_row < out_vec_size) {
      y[row * quads_per_simd] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_fast_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;

      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  // In this case we need to properly guard all our reads because there isn't
  // even 1 tile in the matrix
  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }

  // In this case the last tile is moved back to redo some output values
  else {
    ws += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

// Affine analog of fp_qmv_wide. Weights carry a scale and bias per group, so
// each group is decoded in 8-value sub-chunks (scale * q + bias, registers
// bounded for any group_size) and reused across the vecs_per_tg vectors.
template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>
METAL_FUNC void qmv_wide_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = SIMD_SIZE / k_lanes;
  constexpr int sub = 8; // values per sub-chunk (== bits bytes, byte-aligned)

  typedef float U;

  const short k_lane = simd_lid % k_lanes;
  const short sg_row = simd_lid / k_lanes;

  const int out_row = tid.y * (results_per_simdgroup * num_simdgroups) +
      results_per_simdgroup * simd_gid + sg_row;
  const int vec0 = tid.x * vecs_per_tg;

  const int row = min(out_row, out_vec_size - 1);

  const int in_vec_size_w = in_vec_size * bits / 8; // bytes per weight row
  const int in_vec_size_g = in_vec_size / group_size;
  const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
  const device T* srow = scales + row * in_vec_size_g;
  const device T* brow = biases + row * in_vec_size_g;

  const device T* xv[vecs_per_tg];
  for (int v = 0; v < vecs_per_tg; v++) {
    xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;
  }

  U result[vecs_per_tg] = {0};

  // Each lane reduces a strided subset of the row's groups: decode the group in
  // 8-value sub-chunks and reuse each chunk across the streamed vectors.
  for (int g = k_lane; g < in_vec_size_g; g += k_lanes) {
    U scale = srow[g];
    U bias = brow[g];
#pragma unroll
    for (int sc = 0; sc < group_size / sub; sc++) {
      const int k0 = g * group_size + sc * sub;
      const device uint8_t* wc = wrow + k0 * bits / 8;
      U w_dq[sub];
      dequantize<U, sub, bits>(wc, scale, bias, w_dq);
#pragma unroll
      for (int v = 0; v < vecs_per_tg; v++) {
        const device T* xc = xv[v] + k0;
        U acc = 0;
#pragma unroll
        for (int i = 0; i < sub; i++) {
          acc += static_cast<U>(xc[i]) * w_dq[i];
        }
        result[v] += acc;
      }
    }
  }

  // Reduce each vector's partial over its k_lanes with a shuffle ladder:
  // simd_sum would mix the results_per_simdgroup rows a simdgroup spans.
  for (int v = 0; v < vecs_per_tg; v++) {
    if constexpr (k_lanes >= 32) {
      result[v] += simd_shuffle_down(result[v], 16);
    }
    if constexpr (k_lanes >= 16) {
      result[v] += simd_shuffle_down(result[v], 8);
    }
    if constexpr (k_lanes >= 8) {
      result[v] += simd_shuffle_down(result[v], 4);
    }
    if constexpr (k_lanes >= 4) {
      result[v] += simd_shuffle_down(result[v], 2);
    }
    if constexpr (k_lanes >= 2) {
      result[v] += simd_shuffle_down(result[v], 1);
    }
  }

  if (k_lane == 0 && out_row < out_vec_size) {
    for (int v = 0; v < vecs_per_tg; v++) {
      if (vec0 + v < M) {
        y[(vec0 + v) * out_vec_size + out_row] = static_cast<T>(result[v]);
      }
    }
  }
}

template <typename T, const int group_size, const int bits>
METAL_FUNC void qvm_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int tn = 32 / pack_factor;
  constexpr int block_size = SIMD_SIZE;

  using W_T =
      typename ConditionalType<power_of_2_bits, uint32_t, uint8_t>::type;
  const device W_T* ws = (const device W_T*)w;

  typedef float U;
  typedef struct {
    W_T wi[tn * bytes_per_pack];
  } vec_w;

  thread vec_w w_local;
  thread U result[tn * pack_factor] = {0};
  thread U scale = 1;
  thread U bias = 0;
  thread U x_local = 0;

  // Adjust positions
  const int out_vec_size_w = out_vec_size * bytes_per_pack / pack_factor;
  const int out_vec_size_g = out_vec_size / group_size;
  int out_col = pack_factor * tn * (tid.y * num_simdgroups + simd_gid);
  ws += out_col * bytes_per_pack / pack_factor + simd_lid * out_vec_size_w;
  scales += out_col / group_size + simd_lid * out_vec_size_g;
  biases += out_col / group_size + simd_lid * out_vec_size_g;
  x += tid.x * in_vec_size + simd_lid;
  y += tid.x * out_vec_size + out_col;

  if (out_col >= out_vec_size) {
    return;
  }

  // Loop over in_vec in blocks of block_size
  int remaining = in_vec_size % block_size;
  if (remaining == 0) {
    for (int i = 0; i < in_vec_size; i += block_size) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w*)ws);
      qouter<U, tn * pack_factor, bits>(
          (thread uint8_t*)&w_local, x_local, scale, bias, result);

      x += block_size;
      scales += block_size * out_vec_size_g;
      biases += block_size * out_vec_size_g;
      ws += block_size * out_vec_size_w;
    }
  } else {
    for (int i = block_size; i < in_vec_size; i += block_size) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w*)ws);

      qouter<U, tn * pack_factor, bits>(
          (thread uint8_t*)&w_local, x_local, scale, bias, result);

      x += block_size;
      scales += block_size * out_vec_size_g;
      biases += block_size * out_vec_size_g;
      ws += block_size * out_vec_size_w;
    }
    if (static_cast<int>(simd_lid) < remaining) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w*)ws);
    } else {
      x_local = 0;
      scale = 0;
      bias = 0;
    }
    qouter<U, tn * pack_factor, bits>(
        (thread uint8_t*)&w_local, x_local, scale, bias, result);
  }

// Accumulate in the simdgroup
#pragma clang loop unroll(full)
  for (int k = 0; k < tn * pack_factor; k++) {
    result[k] = simd_sum(result[k]);
  }

  // Store the result
  if (simd_lid == 0) {
#pragma clang loop unroll(full)
    for (int k = 0; k < tn * pack_factor; k++) {
      y[k] = static_cast<T>(result[k]);
    }
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void qmm_t_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void qmm_n_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, false, BK_padded, BN_padded>;
  using loader_x_t = mlx::steel::
      BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 1, 4>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      0,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  auto wl = (const device uint8_t*)w;

  // Set the block
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * bytes_per_pack / pack_factor;
  scales += y_col / group_size;
  biases += y_col / group_size;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, N, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, num_els));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, BM));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device T*& scales,
    const device T*& biases,
    device T*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device T*& scales,
    const device T*& biases,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T*& y,
    int output_stride,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx;
  uint32_t w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <typename T, int group_size, int bits, int D, bool batched>
[[kernel]] void affine_qmv_quad(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmv_quad_impl<T, group_size, bits, D>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      quad_gid,
      quad_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void affine_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmv_fast_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits, bool batched>
[[kernel]] void affine_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmv_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    int vecs_per_tg,
    int k_lanes,
    bool batched>
[[kernel]] void affine_qmv_wide(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmv_wide_impl<T, group_size, bits, vecs_per_tg, k_lanes>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      M,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits, bool batched>
[[kernel]] void affine_qvm(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qvm_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits, int split_k = 32>
[[kernel]] void affine_qvm_split_k(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    const constant int& final_block_size [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      y,
      out_vec_size * M,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);

  // When (in_vec_size % split_k != 0) the final block needs to be smaller
  int in_vec_size_adj =
      tid.z % split_k == split_k - 1 ? final_block_size : in_vec_size;

  qvm_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size_adj,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const bool batched,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_qmm_t(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool batched,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_qmm_n(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }

  qmm_n_impl<T, group_size, bits, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void affine_gather_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmv_fast_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void affine_gather_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmv_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void affine_gather_qvm(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qvm_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_gather_qmm_t(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_gather_qmm_n(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmm_n_impl<T, group_size, bits, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose>
[[kernel]] void affine_gather_qmm_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    const device uint32_t* indices [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      transpose,
      BK_padded,
      transpose ? BK_padded : BN_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;
  biases += transpose ? y_col_long * K_g : y_col / group_size;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Prepare threadgroup mma operation
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    // Prepare threadgroup loading operations
    thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        biases + index * stride_s,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);

    // Matrices are all aligned check nothing
    if (align_M && align_N) {
      gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }

      // Store results to device memory
      if (offset_next - offset == BM) {
        mma_op.store_result(y, N);
      } else {
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      }
    } else {
      // Tile aligned so check outside of the hot loop
      if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
        gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }

        // Store results to device memory
        if (offset_next - offset == BM) {
          mma_op.store_result(y, N);
        } else {
          mma_op.store_result_slice(
              y, N, short2(0, offset), short2(BN, offset_next));
        }
      }

      // Tile partially aligned check rows
      else if (align_N || tgp_bn == BN) {
        gemm_loop_unaligned<false, true, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      }

      // Tile partially aligned check cols
      else if (align_M || tgp_bm == BM) {
        gemm_loop_unaligned<true, false, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(tgp_bn, offset_next));
      }

      // Nothing aligned so check both rows and cols
      else {
        gemm_loop_unaligned<false, false, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(tgp_bn, offset_next));
      }
    }
  }
}

template <typename T, const int group_size, const int bits>
[[kernel]] void affine_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    device T* scales [[buffer(2)]],
    device T* biases [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  constexpr float eps = 1e-7;
  constexpr int simd_size = 32;
  constexpr float n_bins = (1 << bits) - 1;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int values_per_reduce = group_size / simd_size;
  constexpr int writes_per_reduce = pack_factor / values_per_reduce;
  constexpr int writes_per_pack =
      writes_per_reduce > 1 ? 1 : values_per_reduce / pack_factor;
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;

  static_assert(
      group_size % simd_size == 0,
      "Group size must be divisible by simd size.");

  size_t offset = index.x + grid_dim.x * size_t(index.y);
  size_t in_index = offset * values_per_reduce;
  size_t out_index = power_of_2_bits
      ? offset * writes_per_pack
      : offset * bytes_per_pack / writes_per_reduce;

  float w_thread[values_per_reduce];
  float w_min = Limits<T>::max;
  float w_max = 0;

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    float val = w[in_index + i];
    w_thread[i] = val;
    w_min = min(w_min, val);
    w_max = max(w_max, val);
  }

  w_min = simd_min(w_min);
  w_max = simd_max(w_max);

  float scale = max((w_max - w_min) / n_bins, eps);
  bool side = abs(w_min) > abs(w_max);
  scale = side ? scale : -scale;
  float edge = side ? w_min : w_max;
  float q0 = round(edge / scale);
  bool at_zero = q0 == 0.0f;
  scale = at_zero ? scale : edge / q0;
  float bias = at_zero ? 0 : edge;

  // Write out the scales and biases
  size_t gindex = in_index / group_size;
  if (in_index % group_size == 0) {
    scales[gindex] = static_cast<T>(scale);
    biases[gindex] = static_cast<T>(bias);
  }

  using OutType = metal::conditional_t<bits == 5, uint64_t, uint32_t>;
  OutType output = 0;

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    uint8_t val = min(round((w_thread[i] - bias) / scale), n_bins);
    if (bits == 8) {
      output = val;
    } else {
      output |= val << (bits * (i % pack_factor));
    }

    if (pack_factor < values_per_reduce && i % pack_factor == pack_factor - 1) {
      out[out_index + i / pack_factor] = output;
      output = 0;
    } else {
#pragma clang loop unroll(full)
      for (int j = 1; j < writes_per_reduce; j++) {
        uint8_t sval = simd_shuffle_down(val, j);
        output |= static_cast<OutType>(sval)
            << (bits * (j * values_per_reduce + i));
      }
    }
  }
  if (bits == 3 || bits == 6) {
    if (in_index % pack_factor == 0 && out_index % bytes_per_pack == 0) {
      out[out_index] = output & 0xff;
      out[out_index + 1] = (output & 0xff00) >> 8;
      out[out_index + 2] = (output & 0xff0000) >> 16;
    }
  } else if (bits == 5) {
    if (in_index % pack_factor == 0 && out_index % bytes_per_pack == 0) {
      out[out_index] = output & 0xff;
      out[out_index + 1] = (output & 0xff00) >> 8;
      out[out_index + 2] = (output & 0xff0000) >> 16;
      out[out_index + 3] = (output & 0xff000000) >> 24;
      out[out_index + 4] = (output & 0xff00000000) >> 32;
    }
  } else {
    if (writes_per_reduce > 0 && out_index % writes_per_reduce == 0) {
      out[out_index / writes_per_reduce] = output;
    }
  }
}

template <typename T, const int group_size, const int bits>
[[kernel]] void affine_dequantize(
    const device uint8_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    device T* out [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  size_t offset = index.x + grid_dim.x * size_t(index.y);
  size_t oindex = offset * pack_factor;
  size_t gindex = oindex / group_size;
  T scale = scales[gindex];
  T bias = biases[gindex];

  out += oindex;

  if (bits == 3) {
    w += offset * bytes_per_pack;
    out[0] = (w[0] & 0x7) * scale + bias;
    out[1] = ((w[0] & 0x38) >> 3) * scale + bias;
    out[2] = (((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * scale + bias;
    out[3] = ((w[1] & 0xe) >> 1) * scale + bias;
    out[4] = ((w[1] & 0x70) >> 4) * scale + bias;
    out[5] = (((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * scale + bias;
    out[6] = ((w[2] & 0x1c) >> 2) * scale + bias;
    out[7] = ((w[2] & 0xe0) >> 5) * scale + bias;
  } else if (bits == 5) {
    w += offset * bytes_per_pack;
    out[0] = (w[0] & 0x1f) * scale + bias;
    out[1] = (((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * scale + bias;
    out[2] = ((w[1] & 0x7c) >> 2) * scale + bias;
    out[3] = (((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * scale + bias;
    out[4] = (((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * scale + bias;
    out[5] = ((w[3] & 0x3e) >> 1) * scale + bias;
    out[6] = (((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * scale + bias;
    out[7] = ((w[4] & 0xf8) >> 3) * scale + bias;
  } else if (bits == 6) {
    w += offset * bytes_per_pack;
    out[0] = (w[0] & 0x3f) * scale + bias;
    out[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
    out[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
    out[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
  } else {
    uint val = w[offset];
#pragma clang loop unroll(full)
    for (int i = 0; i < pack_factor; i++) {
      uint8_t d;
      if (bits == 2) {
        d = (val >> (bits * i)) & 0x03;
      } else if (bits == 4) {
        d = (val >> (bits * i)) & 0x0f;
      } else if (bits == 8) {
        d = val;
      }
      out[i] = scale * d + bias;
    }
  }
}

// Small-M affine QMM: one 8x8 simdgroup MMA tile covers M <= 8 exactly, so
// each quantized weight group is read and dequantized ONCE and reused by every
// row (qmv_wide re-streams the weights per 5-row tile). K is split across the
// threadgroup's 8 simdgroups (split-K) so the serial loop per simdgroup is
// short. Rows >= M are clamped onto row M-1 and their results never stored.
// Two inner paths:
//  - bits == 4 (direct-fragment): each lane dequantizes exactly the two
//    8x8-B-fragment elements it owns — columns n0+fragCol/fragCol+1, field
//    fragRow of each packed uint32. Every uint32 is requested by the 8 lanes
//    of distinct fragRow at the same unrolled step, so the whole line is
//    used with no threadgroup staging and no barriers; weight loads are two
//    uint4 per column per 64-K block, A rows ride as preloaded packed
//    uint32s.
//  - bits == 8 (staged): dequant into a per-simdgroup threadgroup tile with
//    a software-pipelined register prefetch behind the MMA chain.
// Dispatched for 5 <= M <= 8, group_size 64, bits 4/8, N >= 2048, K % 512 == 0,
// non-batched, bf16 activations. fp32 accumulation. Port of avlp12's qmm_mma4
// (MIT), via jundot/omlx's small_m_qmm.py; the direct-fragment path is ours.
template <typename T, int group_size, int bits>
[[kernel]] void affine_qmm_mma8(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  static_assert(
      group_size == 64, "affine_qmm_mma8 stages one quantization group");
  static_assert(bits == 4 || bits == 8, "affine_qmm_mma8 covers 4/8-bit");
  constexpr int pack = 32 / bits; // values per packed uint32

  const int KPS = K / 8; // K-span per simdgroup (split-K over 8 simdgroups)
  const uint sg = tid >> 5;
  const uint lane = tid & 31;
  const int n0 = (int)tgid.x * 8; // one threadgroup -> 8 output columns

  threadgroup float red[8 * 64]; // cross-simdgroup reduction

  simdgroup_matrix<float, 8, 8> C(0);

  const int kg = K / group_size; // scale/bias row stride
  const int kw = K / pack; // packed-weight row stride (uint32s)
  const uint32_t qmask = (1u << bits) - 1;

  // The 8x8 fragment this lane holds (Metal simdgroup_matrix 8x8 layout,
  // same mapping as conv.metal's winograd transform): row ((lane/4) & 4) +
  // ((lane/2) % 4), columns 2*((lane/4) & 2) + 2*(lane%2) and its +1. The
  // row clamp keeps the A reads in bounds for M < 8.
  const int qid = (int)(lane >> 2);
  const int fragRow = (qid & 4) + (int)((lane >> 1) & 3);
  const int am = min(fragRow, M - 1);
  const int acol = ((qid & 2) << 1) + ((int)(lane & 1) << 1);
  const device T* arow = x + (size_t)am * K;

  const int kbeg = (int)sg * KPS;

  if constexpr (bits == 4) {
    // Direct-fragment path: no staging, no barriers. Lane fragment:
    // B(fragRow, acol), B(fragRow, acol+1) per 8-k step; at 4 bits the
    // kt-th uint32 of a column's 64-k block holds k = block + 8*kt .. +7,
    // so this lane's value is field fragRow of uint32 kt — consecutive,
    // so the block's 8 uint32 load as two uint4.
    const int nA = n0 + acol;
    const bool aValid = nA < N;
    const bool bValid = nA + 1 < N;
    const device uint32_t* wa = w + (size_t)nA * kw;
    const device uint32_t* wb = wa + kw;
    const device T* sa = scales + (size_t)nA * kg;
    const device T* sb = sa + kg;
    const device T* ba = biases + (size_t)nA * kg;
    const device T* bb = ba + kg;
    const uint sh = (uint)(fragRow * bits);

    for (int kk = 0; kk < KPS; kk += 64) {
      const int ka = kbeg + kk;
      const int g = ka >> 6; // quantization group index
      const int woff = ka / pack; // 64 values = 8 uint32 at 4-bit
      uint32_t pa[8] = {0}, pb[8] = {0}, ax[8];
      float s0 = 0, b0 = 0, s1 = 0, b1 = 0;
      if (aValid) {
        s0 = (float)sa[g];
        b0 = (float)ba[g];
        *(thread uint4*)(pa) = *((const device uint4*)(wa + woff));
        *(thread uint4*)(pa + 4) = *((const device uint4*)(wa + woff) + 1);
      }
      if (bValid) {
        s1 = (float)sb[g];
        b1 = (float)bb[g];
        *(thread uint4*)(pb) = *((const device uint4*)(wb + woff));
        *(thread uint4*)(pb + 4) = *((const device uint4*)(wb + woff) + 1);
      }
#pragma unroll
      for (int kt = 0; kt < 8; ++kt) {
        ax[kt] = *(const device uint32_t*)(arow + ka + kt * 8 + acol);
      }
      simdgroup_matrix<T, 8, 8> A, B;
#pragma unroll
      for (int kt = 0; kt < 8; ++kt) {
        A.thread_elements()[0] = as_type<T>((ushort)(ax[kt] & 0xffffu));
        A.thread_elements()[1] = as_type<T>((ushort)(ax[kt] >> 16));
        B.thread_elements()[0] = (T)((float)((pa[kt] >> sh) & qmask) * s0 + b0);
        B.thread_elements()[1] = (T)((float)((pb[kt] >> sh) & qmask) * s1 + b1);
        simdgroup_multiply_accumulate(C, A, B, C);
      }
    }
  } else {
    threadgroup T bs[8 * 512]; // per-simdgroup 64k x 8n dequant stage
    threadgroup T* bt = bs + sg * 512;

    // Software pipeline: the packed weights + scale/bias for iteration kk+64
    // are loaded into registers while the MMA chain for kk runs, hiding the
    // streaming-load latency behind 8 MMAs instead of exposing it per
    // barrier.
    const int j = (int)(lane & 7); // n column this lane dequantizes
    const int kq = (int)(lane >> 3); // k quarter (16 values each)
    const int n = n0 + j;
    const bool nValid = n < N;
    const device uint32_t* wr =
        w + (size_t)n * kw + (kbeg / pack) + kq * (16 / pack);
    const device T* sr = scales + (size_t)n * kg;
    const device T* br = biases + (size_t)n * kg;
    uint32_t pNext[16 / pack];
    float sNext = 0, bNext = 0;
    if (nValid) {
      sNext = (float)sr[kbeg >> 6];
      bNext = (float)br[kbeg >> 6];
#pragma unroll
      for (int u = 0; u < 16 / pack; ++u) {
        pNext[u] = wr[u];
      }
    }
    for (int kk = 0; kk < KPS; kk += 64) {
      const int ka = kbeg + kk; // multiple of 64: one quantization group covers
      const float s = sNext;
      const float bb = bNext;
      uint32_t pCur[16 / pack];
#pragma unroll
      for (int u = 0; u < 16 / pack; ++u) {
        pCur[u] = pNext[u];
      }
      // Issue next iteration's loads before the barrier; consumed after the
      // MMA.
      if (nValid && kk + 64 < KPS) {
        const device uint32_t* wn = wr + (64 / pack);
        sNext = (float)sr[(ka + 64) >> 6];
        bNext = (float)br[(ka + 64) >> 6];
#pragma unroll
        for (int u = 0; u < 16 / pack; ++u) {
          pNext[u] = wn[u];
        }
        wr = wn;
      }
      if (nValid) {
#pragma unroll
        for (int u = 0; u < 16 / pack; ++u) {
          const uint32_t p = pCur[u];
#pragma unroll
          for (int t = 0; t < pack; ++t) {
            bt[(kq * 16 + u * pack + t) * 8 + j] =
                (T)((float)((p >> (bits * t)) & qmask) * s + bb);
          }
        }
      } else {
#pragma unroll
        for (int t = 0; t < 16; ++t) {
          bt[(kq * 16 + t) * 8 + j] = (T)0;
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);

      simdgroup_matrix<T, 8, 8> A, B;
      const device T* aptr = arow + ka;
#pragma unroll
      for (int kt = 0; kt < 8; ++kt) {
        A.thread_elements()[0] = aptr[kt * 8 + acol];
        A.thread_elements()[1] = aptr[kt * 8 + acol + 1];
        simdgroup_load(B, bt + kt * 64, 8);
        simdgroup_multiply_accumulate(C, A, B, C);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  simdgroup_store(C, red + sg * 64, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Sum the 8 split-K partials and write out.
  for (int i = (int)tid; i < 64; i += 256) {
    const int m = i >> 3;
    const int j = i & 7;
    const int n = n0 + j;
    if (m < M && n < N) {
      float v = 0.0f;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        v += red[q * 64 + i];
      }
      y[(size_t)m * N + n] = (T)v;
    }
  }
}

// Small-M affine QMM, 16-wide-N variant of affine_qmm_mma8: one threadgroup
// covers TWO adjacent 8-column tiles that share the A fragments, so each
// 64-k step loads A once, streams four weight columns per lane, and issues
// two MMAs — twice the independent loads in flight per lane (this family is
// latency-bound, not bandwidth-bound) for half the threadgroups and half the
// per-output A traffic and address math. 4-bit direct-fragment only. Rows
// >= M clamp onto row M-1 and their results are never stored. Same dispatch
// window as affine_qmm_mma8 (gated behind MLX_QMM_MMA8_N16 for A/B). fp32
// accumulation, split-K over 8 simdgroups — the accumulation order per
// output matches affine_qmm_mma8's. `full_tiles` (host: N % 16 == 0) drops
// the per-column range guards on the weight/scale loads.
template <typename T, int group_size, int bits, bool full_tiles>
[[kernel]] void affine_qmm_mma8n16(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  static_assert(
      group_size == 64, "affine_qmm_mma8n16 stages one quantization group");
  static_assert(bits == 4, "affine_qmm_mma8n16 is 4-bit direct-fragment only");
  constexpr int pack = 32 / bits; // values per packed uint32

  const int KPS = K / 8; // K-span per simdgroup (split-K over 8 simdgroups)
  const uint sg = tid >> 5;
  const uint lane = tid & 31;
  const int n0 = (int)tgid.x * 16; // one threadgroup -> 16 output columns

  threadgroup float red[8 * 128]; // cross-simdgroup reduction, two n tiles

  simdgroup_matrix<float, 8, 8> C0(0);
  simdgroup_matrix<float, 8, 8> C1(0);

  const int kg = K / group_size; // scale/bias row stride
  const int kw = K / pack; // packed-weight row stride (uint32s)
  const uint32_t qmask = (1u << bits) - 1;

  // Lane fragment mapping: see affine_qmm_mma8. C0 carries columns
  // n0..n0+7, C1 columns n0+8..n0+15; both share the A fragment.
  const int qid = (int)(lane >> 2);
  const int fragRow = (qid & 4) + (int)((lane >> 1) & 3);
  const int am = min(fragRow, M - 1);
  const int acol = ((qid & 2) << 1) + ((int)(lane & 1) << 1);
  const device T* arow = x + (size_t)am * K;

  const int kbeg = (int)sg * KPS;

  const int nA = n0 + acol;
  const int nC = nA + 8;
  const bool aValid = nA < N;
  const bool bValid = nA + 1 < N;
  const bool cValid = nC < N;
  const bool dValid = nC + 1 < N;
  // One packed-weight pointer and one scale/bias pair per lane; the other
  // three columns index off them (row strides kw and kg), and the
  // activation word loads sit inside the kt loop. Same arithmetic as the
  // twelve-pointer form, ~3% less time per launch: the tile is bound by
  // its live registers (see the 2026-09-05 tile study).
  const device uint32_t* wa = w + (size_t)nA * kw;
  const device T* sa = scales + (size_t)nA * kg;
  const device T* ba = biases + (size_t)nA * kg;
  const uint sh = (uint)(fragRow * bits);

  for (int kk = 0; kk < KPS; kk += 64) {
    const int ka = kbeg + kk;
    const int g = ka >> 6; // quantization group index
    const int woff = ka / pack; // 64 values = 8 uint32 at 4-bit
    uint32_t pa[8] = {0}, pb[8] = {0}, pc[8] = {0}, pd[8] = {0}, ax[8];
    float s0 = 0, b0 = 0, s1 = 0, b1 = 0, s2 = 0, b2 = 0, s3 = 0, b3 = 0;
    {
      const device uint4* wq = (const device uint4*)(wa + woff);
      const int kw4 = kw / 4;
      // full_tiles: the host proved N % 16 == 0, so every column of the
      // tile is in range and the four guards below compile out. The
      // predicated load sequence costs ~8% of the kernel's time (the tile
      // is register-bound; the 2026-09-05 tile study).
      if (full_tiles || aValid) {
        s0 = (float)sa[g];
        b0 = (float)ba[g];
        *(thread uint4*)(pa) = wq[0];
        *(thread uint4*)(pa + 4) = wq[1];
      }
      if (full_tiles || bValid) {
        s1 = (float)sa[kg + g];
        b1 = (float)ba[kg + g];
        *(thread uint4*)(pb) = wq[kw4];
        *(thread uint4*)(pb + 4) = wq[kw4 + 1];
      }
      if (full_tiles || cValid) {
        s2 = (float)sa[8 * kg + g];
        b2 = (float)ba[8 * kg + g];
        *(thread uint4*)(pc) = wq[8 * kw4];
        *(thread uint4*)(pc + 4) = wq[8 * kw4 + 1];
      }
      if (full_tiles || dValid) {
        s3 = (float)sa[9 * kg + g];
        b3 = (float)ba[9 * kg + g];
        *(thread uint4*)(pd) = wq[9 * kw4];
        *(thread uint4*)(pd + 4) = wq[9 * kw4 + 1];
      }
    }
    simdgroup_matrix<T, 8, 8> A, B0, B1;
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      ax[kt] = *(const device uint32_t*)(arow + ka + kt * 8 + acol);
      A.thread_elements()[0] = as_type<T>((ushort)(ax[kt] & 0xffffu));
      A.thread_elements()[1] = as_type<T>((ushort)(ax[kt] >> 16));
      B0.thread_elements()[0] = (T)((float)((pa[kt] >> sh) & qmask) * s0 + b0);
      B0.thread_elements()[1] = (T)((float)((pb[kt] >> sh) & qmask) * s1 + b1);
      B1.thread_elements()[0] = (T)((float)((pc[kt] >> sh) & qmask) * s2 + b2);
      B1.thread_elements()[1] = (T)((float)((pd[kt] >> sh) & qmask) * s3 + b3);
      simdgroup_multiply_accumulate(C0, A, B0, C0);
      simdgroup_multiply_accumulate(C1, A, B1, C1);
    }
  }

  simdgroup_store(C0, red + sg * 128, 8);
  simdgroup_store(C1, red + sg * 128 + 64, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Sum the 8 split-K partials and write out (columns n0..n0+7 from the
  // first 64 slots per simdgroup, n0+8..n0+15 from the second).
  for (int i = (int)tid; i < 128; i += 256) {
    const int t = i >> 6; // n tile
    const int m = (i & 63) >> 3;
    const int j = (i & 7) + (t << 3);
    const int n = n0 + j;
    if (m < M && n < N) {
      float v = 0.0f;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        v += red[q * 128 + i];
      }
      y[(size_t)m * N + n] = (T)v;
    }
  }
}

// The 128 + q bit trick for the v2 tile: OR a 4-bit nibble into the low
// mantissa bits of 128.0 and the value is 128 + q * unit exactly (unit is 1
// for bfloat16's 7-bit mantissa, 1/8 for half's 10-bit mantissa).
template <typename T>
struct qmm_v2_base;
template <>
struct qmm_v2_base<bfloat16_t> {
  static constant constexpr const ushort bits = 0x4300;
  static constant constexpr const float inv_unit = 1.0f;
};
template <>
struct qmm_v2_base<half> {
  static constant constexpr const ushort bits = 0x5800;
  static constant constexpr const float inv_unit = 8.0f;
};

// v2 of affine_qmm_mma8n16: scale after accumulate. The MMA multiplies the
// activations by the raw nibbles (as 128 + q via the mantissa trick, one
// bit-op per weight instead of a dequant chain plus a bf16 round) into a
// per-group f32 partial G, and the group's scale/bias fold in once per
// group per output: C += s' * G + (b - 128 s') * rowsum(A), with s' = s /
// unit. Each lane's C fragment elements sit in the very columns whose
// scale/bias it already streams, and rowsum(A) over the group is two lane
// shuffles. The dequantized weight is never rounded to T, so the result is
// closer to the qmv kernels' f32 dequant than affine_qmm_mma8n16's.
// `prefetch` double-buffers the next group's packed weights across the MMA
// step (latency-bound family). Same dispatch window as affine_qmm_mma8n16.
template <typename T, int group_size, int bits, bool prefetch>
[[kernel]] void affine_qmm_mma8n16v2(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  static_assert(
      group_size == 64, "affine_qmm_mma8n16v2 stages one quantization group");
  static_assert(bits == 4, "affine_qmm_mma8n16v2 is 4-bit only");
  constexpr int pack = 32 / bits;
  constexpr ushort qbase = qmm_v2_base<T>::bits;
  constexpr float inv_unit = qmm_v2_base<T>::inv_unit;

  const int KPS = K / 8;
  const uint sg = tid >> 5;
  const uint lane = tid & 31;
  const int n0 = (int)tgid.x * 16;

  threadgroup float red[8 * 128];

  const int kg = K / group_size;
  const int kw = K / pack;

  const int qid = (int)(lane >> 2);
  const int fragRow = (qid & 4) + (int)((lane >> 1) & 3);
  const int am = min(fragRow, M - 1);
  const int acol = ((qid & 2) << 1) + ((int)(lane & 1) << 1);
  const device T* arow = x + (size_t)am * K;

  const int kbeg = (int)sg * KPS;

  // One packed-weight pointer and one scale/bias pair per lane (the other
  // three columns index off them), no per-column range guards — the host
  // dispatches v2 only for N % 16 == 0 — and the activation words load in
  // the kt loop: the same diet as affine_qmm_mma8n16, -6% per launch over
  // the twelve-pointer guarded form (2026-09-05 tile study). `prefetch`
  // is kept for the kernel names only; every load-ahead form lost.
  const int nA = n0 + acol;
  const device uint32_t* wa = w + (size_t)nA * kw;
  const device T* sa = scales + (size_t)nA * kg;
  const device T* ba = biases + (size_t)nA * kg;
  const uint sh = (uint)(fragRow * bits);
  const int kw4 = kw / 4;
  // This lane's four outputs: row fragRow, columns nA, nA+1, nC, nC+1.
  float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

  for (int kk = 0; kk < KPS; kk += 64) {
    const int ka = kbeg + kk;
    const int g = ka >> 6;
    const int woff = ka / pack;
    uint32_t pa[8], pb[8], pc[8], pd[8];
    {
      const device uint4* wq = (const device uint4*)(wa + woff);
      *(thread uint4*)(pa) = wq[0];
      *(thread uint4*)(pa + 4) = wq[1];
      *(thread uint4*)(pb) = wq[kw4];
      *(thread uint4*)(pb + 4) = wq[kw4 + 1];
      *(thread uint4*)(pc) = wq[8 * kw4];
      *(thread uint4*)(pc + 4) = wq[8 * kw4 + 1];
      *(thread uint4*)(pd) = wq[9 * kw4];
      *(thread uint4*)(pd + 4) = wq[9 * kw4 + 1];
    }
    const float s0 = (float)sa[g];
    const float b0 = (float)ba[g];
    const float s1 = (float)sa[kg + g];
    const float b1 = (float)ba[kg + g];
    const float s2 = (float)sa[8 * kg + g];
    const float b2 = (float)ba[8 * kg + g];
    const float s3 = (float)sa[9 * kg + g];
    const float b3 = (float)ba[9 * kg + g];

    simdgroup_matrix<float, 8, 8> G0(0);
    simdgroup_matrix<float, 8, 8> G1(0);
    float rs = 0.0f;
    simdgroup_matrix<T, 8, 8> A, B0, B1;
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      const uint32_t ax = *(const device uint32_t*)(arow + ka + kt * 8 + acol);
      const T a0 = as_type<T>((ushort)(ax & 0xffffu));
      const T a1 = as_type<T>((ushort)(ax >> 16));
      A.thread_elements()[0] = a0;
      A.thread_elements()[1] = a1;
      rs += (float)a0 + (float)a1;
      B0.thread_elements()[0] =
          as_type<T>((ushort)(qbase | ((pa[kt] >> sh) & 0xfu)));
      B0.thread_elements()[1] =
          as_type<T>((ushort)(qbase | ((pb[kt] >> sh) & 0xfu)));
      B1.thread_elements()[0] =
          as_type<T>((ushort)(qbase | ((pc[kt] >> sh) & 0xfu)));
      B1.thread_elements()[1] =
          as_type<T>((ushort)(qbase | ((pd[kt] >> sh) & 0xfu)));
      simdgroup_multiply_accumulate(G0, A, B0, G0);
      simdgroup_multiply_accumulate(G1, A, B1, G1);
    }
    rs += simd_shuffle_xor(rs, (ushort)1);
    rs += simd_shuffle_xor(rs, (ushort)8);

    const float k0 = s0 * inv_unit;
    const float k1 = s1 * inv_unit;
    const float k2 = s2 * inv_unit;
    const float k3 = s3 * inv_unit;
    c00 += k0 * G0.thread_elements()[0] + (b0 - 128.0f * k0) * rs;
    c01 += k1 * G0.thread_elements()[1] + (b1 - 128.0f * k1) * rs;
    c10 += k2 * G1.thread_elements()[0] + (b2 - 128.0f * k2) * rs;
    c11 += k3 * G1.thread_elements()[1] + (b3 - 128.0f * k3) * rs;
  }
#undef QMM_V2_LOAD

  // Stage this lane's C elements at their fragment positions, then sum the
  // 8 split-K partials exactly like affine_qmm_mma8n16.
  {
    threadgroup float* r = red + sg * 128 + fragRow * 8 + acol;
    r[0] = c00;
    r[1] = c01;
    r[64] = c10;
    r[65] = c11;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int i = (int)tid; i < 128; i += 256) {
    const int t = i >> 6;
    const int m = (i & 63) >> 3;
    const int j = (i & 7) + (t << 3);
    const int n = n0 + j;
    if (m < M && n < N) {
      float v = 0.0f;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        v += red[q * 128 + i];
      }
      y[(size_t)m * N + n] = (T)v;
    }
  }
}

// Small-M affine QMM, 32-wide-N variant of affine_qmm_mma8n16: one
// threadgroup covers FOUR adjacent 8-column tiles sharing the A fragments —
// eight weight streams per lane and four MMAs per 64-k step, quadrupling
// the independent loads in flight (latency-bound family) for a quarter of
// the threadgroups. Register-heavy by design: whether the extra ILP beats
// the occupancy loss is decided by the census A/B (opt-in via
// MLX_QMM_MMA8_N32). 4-bit direct-fragment only; rows >= M clamp onto row
// M-1 and are never stored. fp32 accumulation, split-K over 8 simdgroups —
// the accumulation order per output matches affine_qmm_mma8's.
template <typename T, int group_size, int bits>
[[kernel]] void affine_qmm_mma8n32(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  static_assert(
      group_size == 64, "affine_qmm_mma8n32 stages one quantization group");
  static_assert(bits == 4, "affine_qmm_mma8n32 is 4-bit direct-fragment only");
  constexpr int pack = 32 / bits; // values per packed uint32

  const int KPS = K / 8; // K-span per simdgroup (split-K over 8 simdgroups)
  const uint sg = tid >> 5;
  const uint lane = tid & 31;
  const int n0 = (int)tgid.x * 32; // one threadgroup -> 32 output columns

  threadgroup float red[8 * 256]; // cross-simdgroup reduction, four n tiles

  simdgroup_matrix<float, 8, 8> C0(0);
  simdgroup_matrix<float, 8, 8> C1(0);
  simdgroup_matrix<float, 8, 8> C2(0);
  simdgroup_matrix<float, 8, 8> C3(0);

  const int kg = K / group_size; // scale/bias row stride
  const int kw = K / pack; // packed-weight row stride (uint32s)
  const uint32_t qmask = (1u << bits) - 1;

  // Lane fragment mapping: see affine_qmm_mma8. C0 carries columns
  // n0..n0+7, C1 +8, C2 +16, C3 +24; all share the A fragment.
  const int qid = (int)(lane >> 2);
  const int fragRow = (qid & 4) + (int)((lane >> 1) & 3);
  const int am = min(fragRow, M - 1);
  const int acol = ((qid & 2) << 1) + (int)((lane & 1) << 1);
  const device T* arow = x + (size_t)am * K;

  const int kbeg = (int)sg * KPS;

  const int nA = n0 + acol;
  const int nC = nA + 8;
  const int nE = nA + 16;
  const int nG = nA + 24;
  const bool aValid = nA < N;
  const bool bValid = nA + 1 < N;
  const bool cValid = nC < N;
  const bool dValid = nC + 1 < N;
  const bool eValid = nE < N;
  const bool fValid = nE + 1 < N;
  const bool gValid = nG < N;
  const bool hValid = nG + 1 < N;
  const device uint32_t* wa = w + (size_t)nA * kw;
  const device uint32_t* wb = wa + kw;
  const device uint32_t* wc = w + (size_t)nC * kw;
  const device uint32_t* wd = wc + kw;
  const device uint32_t* we = w + (size_t)nE * kw;
  const device uint32_t* wf = we + kw;
  const device uint32_t* wg = w + (size_t)nG * kw;
  const device uint32_t* wh = wg + kw;
  const device T* sa = scales + (size_t)nA * kg;
  const device T* sb = sa + kg;
  const device T* sc = scales + (size_t)nC * kg;
  const device T* sd = sc + kg;
  const device T* ses = scales + (size_t)nE * kg;
  const device T* sfs = ses + kg;
  const device T* sgs = scales + (size_t)nG * kg;
  const device T* shs = sgs + kg;
  const device T* ba = biases + (size_t)nA * kg;
  const device T* bb = ba + kg;
  const device T* bc = biases + (size_t)nC * kg;
  const device T* bd = bc + kg;
  const device T* bes = biases + (size_t)nE * kg;
  const device T* bfs = bes + kg;
  const device T* bgs = biases + (size_t)nG * kg;
  const device T* bhs = bgs + kg;
  const uint sh = (uint)(fragRow * bits);

  for (int kk = 0; kk < KPS; kk += 64) {
    const int ka = kbeg + kk;
    const int g = ka >> 6; // quantization group index
    const int woff = ka / pack; // 64 values = 8 uint32 at 4-bit
    uint32_t pa[8] = {0}, pb[8] = {0}, pc[8] = {0}, pd[8] = {0};
    uint32_t pe[8] = {0}, pf[8] = {0}, pg[8] = {0}, ph[8] = {0};
    uint32_t ax[8];
    float s0 = 0, b0 = 0, s1 = 0, b1 = 0, s2 = 0, b2 = 0, s3 = 0, b3 = 0;
    float s4 = 0, b4 = 0, s5 = 0, b5 = 0, s6 = 0, b6 = 0, s7 = 0, b7 = 0;
    if (aValid) {
      s0 = (float)sa[g];
      b0 = (float)ba[g];
      *(thread uint4*)(pa) = *((const device uint4*)(wa + woff));
      *(thread uint4*)(pa + 4) = *((const device uint4*)(wa + woff) + 1);
    }
    if (bValid) {
      s1 = (float)sb[g];
      b1 = (float)bb[g];
      *(thread uint4*)(pb) = *((const device uint4*)(wb + woff));
      *(thread uint4*)(pb + 4) = *((const device uint4*)(wb + woff) + 1);
    }
    if (cValid) {
      s2 = (float)sc[g];
      b2 = (float)bc[g];
      *(thread uint4*)(pc) = *((const device uint4*)(wc + woff));
      *(thread uint4*)(pc + 4) = *((const device uint4*)(wc + woff) + 1);
    }
    if (dValid) {
      s3 = (float)sd[g];
      b3 = (float)bd[g];
      *(thread uint4*)(pd) = *((const device uint4*)(wd + woff));
      *(thread uint4*)(pd + 4) = *((const device uint4*)(wd + woff) + 1);
    }
    if (eValid) {
      s4 = (float)ses[g];
      b4 = (float)bes[g];
      *(thread uint4*)(pe) = *((const device uint4*)(we + woff));
      *(thread uint4*)(pe + 4) = *((const device uint4*)(we + woff) + 1);
    }
    if (fValid) {
      s5 = (float)sfs[g];
      b5 = (float)bfs[g];
      *(thread uint4*)(pf) = *((const device uint4*)(wf + woff));
      *(thread uint4*)(pf + 4) = *((const device uint4*)(wf + woff) + 1);
    }
    if (gValid) {
      s6 = (float)sgs[g];
      b6 = (float)bgs[g];
      *(thread uint4*)(pg) = *((const device uint4*)(wg + woff));
      *(thread uint4*)(pg + 4) = *((const device uint4*)(wg + woff) + 1);
    }
    if (hValid) {
      s7 = (float)shs[g];
      b7 = (float)bhs[g];
      *(thread uint4*)(ph) = *((const device uint4*)(wh + woff));
      *(thread uint4*)(ph + 4) = *((const device uint4*)(wh + woff) + 1);
    }
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      ax[kt] = *(const device uint32_t*)(arow + ka + kt * 8 + acol);
    }
    simdgroup_matrix<T, 8, 8> A, B0, B1, B2, B3;
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      A.thread_elements()[0] = as_type<T>((ushort)(ax[kt] & 0xffffu));
      A.thread_elements()[1] = as_type<T>((ushort)(ax[kt] >> 16));
      B0.thread_elements()[0] = (T)((float)((pa[kt] >> sh) & qmask) * s0 + b0);
      B0.thread_elements()[1] = (T)((float)((pb[kt] >> sh) & qmask) * s1 + b1);
      B1.thread_elements()[0] = (T)((float)((pc[kt] >> sh) & qmask) * s2 + b2);
      B1.thread_elements()[1] = (T)((float)((pd[kt] >> sh) & qmask) * s3 + b3);
      B2.thread_elements()[0] = (T)((float)((pe[kt] >> sh) & qmask) * s4 + b4);
      B2.thread_elements()[1] = (T)((float)((pf[kt] >> sh) & qmask) * s5 + b5);
      B3.thread_elements()[0] = (T)((float)((pg[kt] >> sh) & qmask) * s6 + b6);
      B3.thread_elements()[1] = (T)((float)((ph[kt] >> sh) & qmask) * s7 + b7);
      simdgroup_multiply_accumulate(C0, A, B0, C0);
      simdgroup_multiply_accumulate(C1, A, B1, C1);
      simdgroup_multiply_accumulate(C2, A, B2, C2);
      simdgroup_multiply_accumulate(C3, A, B3, C3);
    }
  }

  simdgroup_store(C0, red + sg * 256, 8);
  simdgroup_store(C1, red + sg * 256 + 64, 8);
  simdgroup_store(C2, red + sg * 256 + 128, 8);
  simdgroup_store(C3, red + sg * 256 + 192, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Sum the 8 split-K partials and write out (tile t covers columns
  // n0 + t*8 .. n0 + t*8 + 7).
  for (int i = (int)tid; i < 256; i += 256) {
    const int t = i >> 6; // n tile
    const int m = (i & 63) >> 3;
    const int j = (i & 7) + (t << 3);
    const int n = n0 + j;
    if (m < M && n < N) {
      float v = 0.0f;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        v += red[q * 256 + i];
      }
      y[(size_t)m * N + n] = (T)v;
    }
  }
}

// Small-M affine QMM, 16-row variant of affine_qmm_mma8: two stacked 8x8 A
// fragments (rows 0-7 and 8-15) share each dequantized B fragment, so the
// weight stream still runs ONCE and amortizes over up to 16 rows for one
// extra accumulator, A load, and MMA per step. 4-bit direct-fragment only
// (the speculative verify/draft path this serves is 4-bit); 8-bit and other
// modes keep their existing routes. Rows >= M clamp onto row M-1 and their
// results are never stored. Dispatched for 9 <= M <= 16, group_size 64,
// bits 4, N >= 4096, K % 512 == 0, non-batched, bf16 activations. fp32
// accumulation, split-K over 8 simdgroups — the accumulation order per
// output matches affine_qmm_mma8's class.
template <typename T, int group_size, int bits>
[[kernel]] void affine_qmm_mma16(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  static_assert(
      group_size == 64, "affine_qmm_mma16 stages one quantization group");
  static_assert(bits == 4, "affine_qmm_mma16 is 4-bit direct-fragment only");
  constexpr int pack = 32 / bits; // values per packed uint32

  const int KPS = K / 8; // K-span per simdgroup (split-K over 8 simdgroups)
  const uint sg = tid >> 5;
  const uint lane = tid & 31;
  const int n0 = (int)tgid.x * 8; // one threadgroup -> 8 output columns

  threadgroup float red[8 * 128]; // cross-simdgroup reduction, 16 rows

  simdgroup_matrix<float, 8, 8> C0(0);
  simdgroup_matrix<float, 8, 8> C1(0);

  const int kg = K / group_size; // scale/bias row stride
  const int kw = K / pack; // packed-weight row stride (uint32s)
  const uint32_t qmask = (1u << bits) - 1;

  // Lane fragment mapping: see affine_qmm_mma8. A0 carries rows 0-7, A1
  // rows 8-15; both clamp onto row M-1 so the reads stay in bounds.
  const int qid = (int)(lane >> 2);
  const int fragRow = (qid & 4) + (int)((lane >> 1) & 3);
  const int am0 = min(fragRow, M - 1);
  const int am1 = min(8 + fragRow, M - 1);
  const int acol = ((qid & 2) << 1) + ((int)(lane & 1) << 1);
  const device T* arow0 = x + (size_t)am0 * K;
  const device T* arow1 = x + (size_t)am1 * K;

  const int kbeg = (int)sg * KPS;

  const int nA = n0 + acol;
  const bool aValid = nA < N;
  const bool bValid = nA + 1 < N;
  const device uint32_t* wa = w + (size_t)nA * kw;
  const device uint32_t* wb = wa + kw;
  const device T* sa = scales + (size_t)nA * kg;
  const device T* sb = sa + kg;
  const device T* ba = biases + (size_t)nA * kg;
  const device T* bb = ba + kg;
  const uint sh = (uint)(fragRow * bits);

  for (int kk = 0; kk < KPS; kk += 64) {
    const int ka = kbeg + kk;
    const int g = ka >> 6; // quantization group index
    const int woff = ka / pack; // 64 values = 8 uint32 at 4-bit
    uint32_t pa[8] = {0}, pb[8] = {0}, ax0[8], ax1[8];
    float s0 = 0, b0 = 0, s1 = 0, b1 = 0;
    if (aValid) {
      s0 = (float)sa[g];
      b0 = (float)ba[g];
      *(thread uint4*)(pa) = *((const device uint4*)(wa + woff));
      *(thread uint4*)(pa + 4) = *((const device uint4*)(wa + woff) + 1);
    }
    if (bValid) {
      s1 = (float)sb[g];
      b1 = (float)bb[g];
      *(thread uint4*)(pb) = *((const device uint4*)(wb + woff));
      *(thread uint4*)(pb + 4) = *((const device uint4*)(wb + woff) + 1);
    }
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      ax0[kt] = *(const device uint32_t*)(arow0 + ka + kt * 8 + acol);
      ax1[kt] = *(const device uint32_t*)(arow1 + ka + kt * 8 + acol);
    }
    simdgroup_matrix<T, 8, 8> A0, A1, B;
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      A0.thread_elements()[0] = as_type<T>((ushort)(ax0[kt] & 0xffffu));
      A0.thread_elements()[1] = as_type<T>((ushort)(ax0[kt] >> 16));
      A1.thread_elements()[0] = as_type<T>((ushort)(ax1[kt] & 0xffffu));
      A1.thread_elements()[1] = as_type<T>((ushort)(ax1[kt] >> 16));
      B.thread_elements()[0] = (T)((float)((pa[kt] >> sh) & qmask) * s0 + b0);
      B.thread_elements()[1] = (T)((float)((pb[kt] >> sh) & qmask) * s1 + b1);
      simdgroup_multiply_accumulate(C0, A0, B, C0);
      simdgroup_multiply_accumulate(C1, A1, B, C1);
    }
  }

  simdgroup_store(C0, red + sg * 128, 8);
  simdgroup_store(C1, red + sg * 128 + 64, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Sum the 8 split-K partials and write out (rows 0-7 then 8-15).
  for (int i = (int)tid; i < 128; i += 256) {
    const int m = i >> 3;
    const int j = i & 7;
    const int n = n0 + j;
    if (m < M && n < N) {
      float v = 0.0f;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        v += red[q * 128 + i];
      }
      y[(size_t)m * N + n] = (T)v;
    }
  }
}
