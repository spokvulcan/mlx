// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

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
