/*
 * Copyright (c) LAIX, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "skylark/inference/blas/ActivationsFP32UKernels.h"

namespace fbgemm {

// tanh is modified from https://github.com/Microsoft/onnxruntime/commit/47551da99451a1f20b34bf83e06c6ccbd638fe13
void __attribute__((noinline)) activationkernel_avx256_tanh_1x1(TanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // TanhConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 32]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add r12, 32\t\n"
      "add r10, 32\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_tanh_2x1(TanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // TanhConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 32]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_tanh_3x1(TanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // TanhConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 32]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_tanh_4x1(TanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // TanhConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 32]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_tanh_5x1(TanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // TanhConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 32]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_tanh_6x1(TanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // TanhConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 32]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_tanh_7x1(TanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // TanhConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 32]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_tanh_8x1(TanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // TanhConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 32]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 40]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}
// sigmoid is modified from https://github.com/Microsoft/onnxruntime/commit/47551da99451a1f20b34bf83e06c6ccbd638fe13
void __attribute__((noinline)) activationkernel_avx256_sigmoid_1x1(SigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // SigmoidConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 40]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add r12, 32\t\n"
      "add r10, 32\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_sigmoid_2x1(SigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // SigmoidConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 40]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_sigmoid_3x1(SigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // SigmoidConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 40]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_sigmoid_4x1(SigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // SigmoidConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 40]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_sigmoid_5x1(SigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // SigmoidConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 40]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_sigmoid_6x1(SigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // SigmoidConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 40]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_sigmoid_7x1(SigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // SigmoidConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 40]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_sigmoid_8x1(SigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // SigmoidConstants
      "mov rcx, [r14 + 48]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vbroadcastss ymm4,DWORD PTR [rcx + 0]\t\n"
      "vbroadcastss ymm5,DWORD PTR [rcx + 4]\t\n"
      "vbroadcastss ymm6,DWORD PTR [rcx + 8]\t\n"
      "vbroadcastss ymm7,DWORD PTR [rcx + 12]\t\n"
      "vbroadcastss ymm8,DWORD PTR [rcx + 16]\t\n"
      "vbroadcastss ymm9,DWORD PTR [rcx + 20]\t\n"
      "vbroadcastss ymm10,DWORD PTR [rcx + 24]\t\n"
      "vbroadcastss ymm11,DWORD PTR [rcx + 28]\t\n"
      "vbroadcastss ymm12,DWORD PTR [rcx + 36]\t\n"
      "vbroadcastss ymm13,DWORD PTR [rcx + 40]\t\n"
      "vbroadcastss ymm14,DWORD PTR [rcx + 44]\t\n"
      "vbroadcastss ymm15,DWORD PTR [rcx + 48]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmaxps ymm0,ymm4,YMMWORD PTR [r12 + 0]   # clamp lower bound\t\n"
      "vmovaps ymm2,ymm7\t\n"
      "vminps  ymm0,ymm5,ymm0                  # clamp upper bound\t\n"
      "vmulps  ymm1,ymm0,ymm0                  # x2\t\n"
      "vbroadcastss ymm3,DWORD PTR [rcx + 32]\t\n"
      "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7\t\n"
      "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5\t\n"
      "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3\t\n"
      "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1\t\n"
      "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8\t\n"
      "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6\t\n"
      "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4\t\n"
      "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2\t\n"
      "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0\t\n"
      "vmulps  ymm2,ymm0,ymm2                  # p = x * p\t\n"
      "vbroadcastss ymm0,DWORD PTR [rcx + 52]\t\n"
      "vdivps  ymm2,ymm2,ymm3\t\n"
      "vxorps  ymm3,ymm3,ymm3\t\n"
      "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5\t\n"
      "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"

      "add rax, 32\t\n"
      "add rdx, 32\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 1\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}
void __attribute__((noinline)) activationkernel_avx256_relu_1x4(ReLUParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vxorps ymm15,ymm15,ymm15\t\n"
      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"

      "add r12, 128\t\n"
      "add r10, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_relu_2x4(ReLUParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vxorps ymm15,ymm15,ymm15\t\n"
      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"

      "add rax, 128\t\n"
      "add rdx, 128\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 4\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_relu_3x4(ReLUParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vxorps ymm15,ymm15,ymm15\t\n"
      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"

      "add rax, 128\t\n"
      "add rdx, 128\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 4\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_relu_4x4(ReLUParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vxorps ymm15,ymm15,ymm15\t\n"
      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"

      "add rax, 128\t\n"
      "add rdx, 128\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 4\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_relu_5x4(ReLUParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vxorps ymm15,ymm15,ymm15\t\n"
      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"

      "add rax, 128\t\n"
      "add rdx, 128\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 4\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_relu_6x4(ReLUParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vxorps ymm15,ymm15,ymm15\t\n"
      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"

      "add rax, 128\t\n"
      "add rdx, 128\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 4\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_relu_7x4(ReLUParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vxorps ymm15,ymm15,ymm15\t\n"
      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"

      "add rax, 128\t\n"
      "add rdx, 128\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 4\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

void __attribute__((noinline)) activationkernel_avx256_relu_8x4(ReLUParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // m
      "mov r8, [r14 + 0]\t\n"
      // b_block_cols
      "mov r9, [r14 + 8]\t\n"
      // Z
      "mov r10, [r14 + 16]\t\n"
      // ldz
      "mov r11, [r14 + 24]\t\n"
      // A
      "mov r12, [r14 + 32]\t\n"
      // lda
      "mov r13, [r14 + 40]\t\n"
      // Make copies of Z and A
      "mov rdx, r10\t\n"
      "mov rax, r12\t\n"

      "vxorps ymm15,ymm15,ymm15\t\n"
      "mov r14, 0\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"
      "add r12, r13\t\n"
      "add r10, r11\t\n"

      "vmovups ymm0,YMMWORD PTR [r12 + 0]\t\n"
      "vmaxps ymm0,ymm15,ymm0\t\n"
      "vmovups YMMWORD PTR [r10 + 0], ymm0\t\n"
      "vmovups ymm1,YMMWORD PTR [r12 + 32]\t\n"
      "vmaxps ymm1,ymm15,ymm1\t\n"
      "vmovups YMMWORD PTR [r10 + 32], ymm1\t\n"
      "vmovups ymm2,YMMWORD PTR [r12 + 64]\t\n"
      "vmaxps ymm2,ymm15,ymm2\t\n"
      "vmovups YMMWORD PTR [r10 + 64], ymm2\t\n"
      "vmovups ymm3,YMMWORD PTR [r12 + 96]\t\n"
      "vmaxps ymm3,ymm15,ymm3\t\n"
      "vmovups YMMWORD PTR [r10 + 96], ymm3\t\n"

      "add rax, 128\t\n"
      "add rdx, 128\t\n"
      "mov r12, rax\t\n"
      "mov r10, rdx\t\n"

      "add r14, 4\t\n"
      "cmp r14, r9\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "rax",
        "rdx",
        "memory");
}

namespace wavernn {

void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_1x4(CoarseSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_2x4(CoarseSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "sub rsi, r9\t\n"
      "sub rcx, r10\t\n"
      "sub rax, r11\t\n"
      "sub rbx, r12\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_3x4(CoarseSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 2\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 2\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 2\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 2\t\n"
      "sub rbx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_4x4(CoarseSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 3\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 3\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 3\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 3\t\n"
      "sub rbx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_5x4(CoarseSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 4\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 4\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 4\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 4\t\n"
      "sub rbx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_6x4(CoarseSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 5\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 5\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 5\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 5\t\n"
      "sub rbx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_7x4(CoarseSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 6\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 6\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 6\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 6\t\n"
      "sub rbx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_sigmoid_addition_8x4(CoarseSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 7\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 7\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 7\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 7\t\n"
      "sub rbx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}
void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_1x4(CoarseTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_2x4(CoarseTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "sub rsi, r9\t\n"
      "sub rcx, r10\t\n"
      "sub rax, r11\t\n"
      "sub rbx, r12\t\n"
      "sub rdx, r13\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_3x4(CoarseTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 2\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 2\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 2\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 2\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 2\t\n"
      "sub rdx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_4x4(CoarseTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 3\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 3\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 3\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 3\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 3\t\n"
      "sub rdx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_5x4(CoarseTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 4\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 4\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 4\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 4\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 4\t\n"
      "sub rdx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_6x4(CoarseTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 5\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 5\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 5\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 5\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 5\t\n"
      "sub rdx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_7x4(CoarseTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 6\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 6\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 6\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 6\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 6\t\n"
      "sub rdx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_coarse_tanh_addition_8x4(CoarseTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 7\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 7\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 7\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 7\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 7\t\n"
      "sub rdx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}
void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_1x4(FineSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_2x4(FineSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "sub rsi, r9\t\n"
      "sub rcx, r10\t\n"
      "sub rax, r11\t\n"
      "sub rbx, r12\t\n"
      "sub rdx, r13\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_3x4(FineSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 2\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 2\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 2\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 2\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 2\t\n"
      "sub rdx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_4x4(FineSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 3\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 3\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 3\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 3\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 3\t\n"
      "sub rdx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_5x4(FineSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 4\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 4\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 4\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 4\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 4\t\n"
      "sub rdx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_6x4(FineSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 5\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 5\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 5\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 5\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 5\t\n"
      "sub rdx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_7x4(FineSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 6\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 6\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 6\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 6\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 6\t\n"
      "sub rdx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_sigmoid_addition_8x4(FineSigmoidParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm15,ymm14\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm15,ymm14\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm15,ymm14\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vaddps ymm13,ymm15,ymm14\t\n"
      "vmovups ymm15,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm15,ymm14\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 7\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 7\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 7\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 7\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 7\t\n"
      "sub rdx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}
void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_1x4(FineTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"
      // T
      "mov rdi, [r14 + 96]\t\n"
      // ldt
      "mov rbp, [r14 + 104]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"
      "add rdi, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_2x4(FineTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"
      // T
      "mov rdi, [r14 + 96]\t\n"
      // ldt
      "mov rbp, [r14 + 104]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "sub rsi, r9\t\n"
      "sub rcx, r10\t\n"
      "sub rax, r11\t\n"
      "sub rbx, r12\t\n"
      "sub rdx, r13\t\n"
      "sub rdi, rbp\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"
      "add rdi, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_3x4(FineTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"
      // T
      "mov rdi, [r14 + 96]\t\n"
      // ldt
      "mov rbp, [r14 + 104]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 2\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 2\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 2\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 2\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 2\t\n"
      "sub rdx, r15\t\n"
      "imul r15, rbp, 2\t\n"
      "sub rdi, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"
      "add rdi, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_4x4(FineTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"
      // T
      "mov rdi, [r14 + 96]\t\n"
      // ldt
      "mov rbp, [r14 + 104]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 3\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 3\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 3\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 3\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 3\t\n"
      "sub rdx, r15\t\n"
      "imul r15, rbp, 3\t\n"
      "sub rdi, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"
      "add rdi, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_5x4(FineTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"
      // T
      "mov rdi, [r14 + 96]\t\n"
      // ldt
      "mov rbp, [r14 + 104]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 4\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 4\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 4\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 4\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 4\t\n"
      "sub rdx, r15\t\n"
      "imul r15, rbp, 4\t\n"
      "sub rdi, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"
      "add rdi, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_6x4(FineTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"
      // T
      "mov rdi, [r14 + 96]\t\n"
      // ldt
      "mov rbp, [r14 + 104]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 5\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 5\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 5\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 5\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 5\t\n"
      "sub rdx, r15\t\n"
      "imul r15, rbp, 5\t\n"
      "sub rdi, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"
      "add rdi, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_7x4(FineTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"
      // T
      "mov rdi, [r14 + 96]\t\n"
      // ldt
      "mov rbp, [r14 + 104]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 6\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 6\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 6\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 6\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 6\t\n"
      "sub rdx, r15\t\n"
      "imul r15, rbp, 6\t\n"
      "sub rdi, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"
      "add rdi, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_fine_tanh_addition_8x4(FineTanhParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // C
      "mov rcx, [r14 + 32]\t\n"
      // ldc
      "mov r10, [r14 + 40]\t\n"
      // A
      "mov rax, [r14 + 48]\t\n"
      // lda
      "mov r11, [r14 + 56]\t\n"
      // B
      "mov rbx, [r14 + 64]\t\n"
      // ldb
      "mov r12, [r14 + 72]\t\n"
      // D
      "mov rdx, [r14 + 80]\t\n"
      // ldd
      "mov r13, [r14 + 88]\t\n"
      // T
      "mov rdi, [r14 + 96]\t\n"
      // ldt
      "mov rbp, [r14 + 104]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "add rdx, r13\t\n"
      "add rdi, rbp\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 0]\t\n"
      "vaddps ymm0,ymm0,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 32]\t\n"
      "vaddps ymm1,ymm1,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vfmadd231ps ymm1,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 64]\t\n"
      "vaddps ymm2,ymm2,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vmovups ymm13,YMMWORD PTR [rdx + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm13,YMMWORD PTR [rdi + 96]\t\n"
      "vaddps ymm3,ymm3,ymm13\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vfmadd231ps ymm3,ymm15,ymm14\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 7\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 7\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 7\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 7\t\n"
      "sub rbx, r15\t\n"
      "imul r15, r13, 7\t\n"
      "sub rdx, r15\t\n"
      "imul r15, rbp, 7\t\n"
      "sub rdi, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"
      "add rdx, 128\t\n"
      "add rdi, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}
void __attribute__((noinline)) wavernnkernel_avx256_hidden_1x4(HiddenParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // A
      "mov rax, [r14 + 32]\t\n"
      // lda
      "mov r11, [r14 + 40]\t\n"
      // B
      "mov rbx, [r14 + 48]\t\n"
      // ldb
      "mov r12, [r14 + 56]\t\n"
      // C
      "mov rcx, [r14 + 64]\t\n"
      // ldc
      "mov r10, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_hidden_2x4(HiddenParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // A
      "mov rax, [r14 + 32]\t\n"
      // lda
      "mov r11, [r14 + 40]\t\n"
      // B
      "mov rbx, [r14 + 48]\t\n"
      // ldb
      "mov r12, [r14 + 56]\t\n"
      // C
      "mov rcx, [r14 + 64]\t\n"
      // ldc
      "mov r10, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "sub rsi, r9\t\n"
      "sub rcx, r10\t\n"
      "sub rax, r11\t\n"
      "sub rbx, r12\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_hidden_3x4(HiddenParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // A
      "mov rax, [r14 + 32]\t\n"
      // lda
      "mov r11, [r14 + 40]\t\n"
      // B
      "mov rbx, [r14 + 48]\t\n"
      // ldb
      "mov r12, [r14 + 56]\t\n"
      // C
      "mov rcx, [r14 + 64]\t\n"
      // ldc
      "mov r10, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 2\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 2\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 2\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 2\t\n"
      "sub rbx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_hidden_4x4(HiddenParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // A
      "mov rax, [r14 + 32]\t\n"
      // lda
      "mov r11, [r14 + 40]\t\n"
      // B
      "mov rbx, [r14 + 48]\t\n"
      // ldb
      "mov r12, [r14 + 56]\t\n"
      // C
      "mov rcx, [r14 + 64]\t\n"
      // ldc
      "mov r10, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 3\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 3\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 3\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 3\t\n"
      "sub rbx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_hidden_5x4(HiddenParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // A
      "mov rax, [r14 + 32]\t\n"
      // lda
      "mov r11, [r14 + 40]\t\n"
      // B
      "mov rbx, [r14 + 48]\t\n"
      // ldb
      "mov r12, [r14 + 56]\t\n"
      // C
      "mov rcx, [r14 + 64]\t\n"
      // ldc
      "mov r10, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 4\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 4\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 4\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 4\t\n"
      "sub rbx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_hidden_6x4(HiddenParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // A
      "mov rax, [r14 + 32]\t\n"
      // lda
      "mov r11, [r14 + 40]\t\n"
      // B
      "mov rbx, [r14 + 48]\t\n"
      // ldb
      "mov r12, [r14 + 56]\t\n"
      // C
      "mov rcx, [r14 + 64]\t\n"
      // ldc
      "mov r10, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 5\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 5\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 5\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 5\t\n"
      "sub rbx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_hidden_7x4(HiddenParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // A
      "mov rax, [r14 + 32]\t\n"
      // lda
      "mov r11, [r14 + 40]\t\n"
      // B
      "mov rbx, [r14 + 48]\t\n"
      // ldb
      "mov r12, [r14 + 56]\t\n"
      // C
      "mov rcx, [r14 + 64]\t\n"
      // ldc
      "mov r10, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 6\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 6\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 6\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 6\t\n"
      "sub rbx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

void __attribute__((noinline)) wavernnkernel_avx256_hidden_8x4(HiddenParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // b_block_cols
      "mov r8, [r14 + 8]\t\n"
      // Z
      "mov rsi, [r14 + 16]\t\n"
      // ldz
      "mov r9, [r14 + 24]\t\n"
      // A
      "mov rax, [r14 + 32]\t\n"
      // lda
      "mov r11, [r14 + 40]\t\n"
      // B
      "mov rbx, [r14 + 48]\t\n"
      // ldb
      "mov r12, [r14 + 56]\t\n"
      // C
      "mov rcx, [r14 + 64]\t\n"
      // ldc
      "mov r10, [r14 + 72]\t\n"

      "mov r14, 0\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "loop_inner%=:\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "add rsi, r9\t\n"
      "add rcx, r10\t\n"
      "add rax, r11\t\n"
      "add rbx, r12\t\n"
      "vmovups ymm15,YMMWORD PTR [rax + 0]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 0]\t\n"
      "vmovups ymm0,YMMWORD PTR [rcx + 0]\t\n"
      "vsubps ymm13,ymm14,ymm0\t\n"
      "vfmadd231ps ymm0,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 0], ymm0\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 32]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 32]\t\n"
      "vmovups ymm1,YMMWORD PTR [rcx + 32]\t\n"
      "vsubps ymm13,ymm14,ymm1\t\n"
      "vfmadd231ps ymm1,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 32], ymm1\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 64]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 64]\t\n"
      "vmovups ymm2,YMMWORD PTR [rcx + 64]\t\n"
      "vsubps ymm13,ymm14,ymm2\t\n"
      "vfmadd231ps ymm2,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 64], ymm2\t\n"

      "vmovups ymm15,YMMWORD PTR [rax + 96]\t\n"
      "vmovups ymm14,YMMWORD PTR [rbx + 96]\t\n"
      "vmovups ymm3,YMMWORD PTR [rcx + 96]\t\n"
      "vsubps ymm13,ymm14,ymm3\t\n"
      "vfmadd231ps ymm3,ymm15,ymm13\t\n"
      "vmovups YMMWORD PTR [rsi + 96], ymm3\t\n"

      "imul r15, r9, 7\t\n"
      "sub rsi, r15\t\n"
      "imul r15, r10, 7\t\n"
      "sub rcx, r15\t\n"
      "imul r15, r11, 7\t\n"
      "sub rax, r15\t\n"
      "imul r15, r12, 7\t\n"
      "sub rbx, r15\t\n"

      "add rsi, 128\t\n"
      "add rax, 128\t\n"
      "add rbx, 128\t\n"
      "add rcx, 128\t\n"

      "add r14, 4\t\n"
      "cmp r14, r8\t\n"
      "jge L_exit%=\t\n"
      "jmp loop_inner%=\t\n"

      "L_exit%=:\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "memory");
}

}  // namespace wavernn

}  // namespace fbgemm
