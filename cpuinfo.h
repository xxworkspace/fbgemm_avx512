#ifndef CPUINFO_H
#define CPUINFO_H

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include <stdint.h>
#include "skylark/inference/blas/cpuid_info.h"

static inline bool cpuinfo_has_x86_avx512f(void) {
       	return (fbgemm::CPUIDInfo::GetCPUIDInfo()).HasAVX512f();
}

static inline bool cpuinfo_has_x86_avx2(void) {
	return (fbgemm::CPUIDInfo::GetCPUIDInfo()).HasAVX2();
}

static inline bool cpuinfo_has_x86_f16c(void) {
	return (fbgemm::CPUIDInfo::GetCPUIDInfo()).HasF16C();
}

static inline bool cpuinfo_has_x86_fma3(void) {
	return true;
}

#endif /* CPUINFO_H */
