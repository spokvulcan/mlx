// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

#include <cstdlib>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

namespace mlx::core::gpu {

// Tesseract diagnostic: MLX_KERNEL_PROFILE=1 serializes every primitive
// into its own command buffer and accumulates GPU time (GPUEndTime -
// GPUStartTime) by "<window> <primitive> <shapes>" key, windowed by the
// value of MLX_KERNEL_PROFILE_ACTIVE (unset = not recorded). Dumped to
// stderr at exit. Perturbs wall time heavily; GPU per-kernel time is the
// signal. Never enable in production.
namespace {
struct KernelProfileEntry {
  long count = 0;
  double gpu_seconds = 0;
};
std::mutex g_kernel_profile_mutex;
std::map<std::string, KernelProfileEntry>* g_kernel_profile = nullptr;

void kernel_profile_dump() {
  std::lock_guard<std::mutex> lock(g_kernel_profile_mutex);
  if (g_kernel_profile == nullptr) {
    return;
  }
  double total = 0;
  long n = 0;
  for (auto& kv : *g_kernel_profile) {
    total += kv.second.gpu_seconds;
    n += kv.second.count;
  }
  fprintf(stderr, "[kernel-profile] total_ms=%.3f count=%ld\n", total * 1e3, n);
  for (auto& kv : *g_kernel_profile) {
    fprintf(
        stderr,
        "[kernel-profile] %.4f %ld %s\n",
        kv.second.gpu_seconds * 1e3,
        kv.second.count,
        kv.first.c_str());
  }
}

bool kernel_profile_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_KERNEL_PROFILE");
    if (env != nullptr && env[0] == '1') {
      std::atexit(kernel_profile_dump);
      return true;
    }
    return false;
  }();
  return enabled;
}

std::string kernel_profile_shape(const array& a) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < a.shape().size(); ++i) {
    if (i) os << ",";
    os << a.shape(i);
  }
  os << "]";
  return os.str();
}
} // namespace

void new_stream(Stream stream) {
  if (stream.device == mlx::core::Device::gpu) {
    metal::device(stream.device).new_queue(stream.index);
  }
}

inline void check_error(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cbuf->error()->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& d = metal::device(s.device);
  auto command_buffer = d.get_command_buffer(s.index);

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(command_buffer, arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  // Retain this op's input buffers until the active command buffer
  // completes (flushed as a single handler at commit; buffers an
  // input donated to the output are held by the output array itself and
  // are not retained here — same lifetime semantics as the old set).
  auto out_data = arr.data_shared_ptr();
  for (auto& in : arr.inputs()) {
    auto p = in.data_shared_ptr();
    if (p != out_data) {
      d.retain_until_commit(s.index, std::move(p));
    }
  }
  for (auto& sb : arr.siblings()) {
    auto p = sb.data_shared_ptr();
    if (p != out_data) {
      d.retain_until_commit(s.index, std::move(p));
    }
  }

  if (kernel_profile_enabled()) {
    const char* active = std::getenv("MLX_KERNEL_PROFILE_ACTIVE");
    if (active != nullptr && active[0] != '\0') {
      std::string key = active;
      key += " ";
      key += arr.primitive().name();
      key += " out" + kernel_profile_shape(arr);
      {
        std::ostringstream os;
        os << arr.dtype();
        key += " " + os.str();
      }
      int shown = 0;
      for (auto& in : arr.inputs()) {
        if (shown++ == 3) break;
        key += " in" + kernel_profile_shape(in);
      }
      d.end_encoding(s.index);
      auto cb = d.get_command_buffer(s.index);
      cb->retain();
      d.commit_command_buffer(s.index);
      cb->waitUntilCompleted();
      double gpu = cb->GPUEndTime() - cb->GPUStartTime();
      check_error(cb);
      cb->release();
      d.get_command_buffer(s.index);
      std::lock_guard<std::mutex> lock(g_kernel_profile_mutex);
      if (g_kernel_profile == nullptr) {
        g_kernel_profile = new std::map<std::string, KernelProfileEntry>();
      }
      auto& e = (*g_kernel_profile)[key];
      e.count += 1;
      e.gpu_seconds += gpu;
      return;
    }
  }

  if (d.command_buffer_needs_commit(s.index)) {
    d.end_encoding(s.index);
    scheduler::notify_new_task(s);
    command_buffer->addCompletedHandler(
        [s](MTL::CommandBuffer* cbuf) {
          scheduler::notify_task_completion(s);
          check_error(cbuf);
        });
    d.commit_command_buffer(s.index);
    d.get_command_buffer(s.index);
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  auto cb = d.get_command_buffer(s.index);
  d.end_encoding(s.index);
  cb->addCompletedHandler([](MTL::CommandBuffer* cbuf) { check_error(cbuf); });
  d.commit_command_buffer(s.index);
  d.get_command_buffer(s.index);
}

void synchronize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  auto cb = d.get_command_buffer(s.index);
  cb->retain();
  d.end_encoding(s.index);
  d.commit_command_buffer(s.index);
  cb->waitUntilCompleted();
  check_error(cb);
  cb->release();
}

} // namespace mlx::core::gpu
