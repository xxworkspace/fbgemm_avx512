// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// From https://github.com/Microsoft/onnxruntime/tree/master/onnxruntime/core/common
#pragma once


namespace fbgemm{

class CPUIDInfo {
public:
  static CPUIDInfo& GetCPUIDInfo() {
    static CPUIDInfo cpuid_info;
    return cpuid_info;
  }

  void EnableAVX512f(){
    has_avx512f_ = true;
  }
  void DisableAVX512f(){
    has_avx512f_ = false;
  }

  bool HasAVX2() const { return has_avx2_; }
  bool HasAVX512f() const { return has_avx512f_; }
  bool HasF16C() const { return has_f16c_; }

private:
  CPUIDInfo() noexcept;
  bool has_avx2_{false};
  bool has_avx512f_{false};
  bool has_f16c_{false};
};

}  // namespace onnxruntime
