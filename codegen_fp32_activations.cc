/*
 * Copyright (c) LAIX, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <array>
#include <cpuid.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "skylark/inference/blas/logging.h"

using namespace std;

void addi(ofstream &of, string i, bool disable = false) {
  if (disable == false)
    of << "      \"" + i + "\\t\\n\"" + "\n";
}

enum ACTIVATIONS {
  RELU = 0,
  SIGMOID = 1,
  TANH = 2,
  COARSE_SIGMOID_ADDITION = 3,
  COARSE_TANH_ADDITION = 4,
  FINE_SIGMOID_ADDITION = 5,
  FINE_TANH_ADDITION = 6,
  HIDDEN = 7
};

struct ISA {
  unsigned avx; // 1, 2 or 3
  string name;
  enum ACTIVATIONS activation;
  vector<vector<unsigned>> shapes;
};

int main() {
  bool iaca = false;

  int eax, ebx, ecx, edx;
  __cpuid(1 /* ecx = vendor string */, eax, ebx, ecx, edx);

  string comma = ",";

  vector<ISA> isa = {
      {2,
       "tanh",
       TANH,
       {
           {1, 1, 0},
           {2, 1, 0},
           {3, 1, 0},
           {4, 1, 0},
           {5, 1, 0},
           {6, 1, 0},
           {7, 1, 0},
           {8, 1, 0},
       }},
      {2,
       "sigmoid",
       SIGMOID,
       {
           {1, 1, 0},
           {2, 1, 0},
           {3, 1, 0},
           {4, 1, 0},
           {5, 1, 0},
           {6, 1, 0},
           {7, 1, 0},
           {8, 1, 0},
       }},
      {2,
       "relu",
       RELU,
       {
           {1, 4, 0},
           {2, 4, 0},
           {3, 4, 0},
           {4, 4, 0},
           {5, 4, 0},
           {6, 4, 0},
           {7, 4, 0},
           {8, 4, 0},
       }},
      {2,
       "coarse_sigmoid_addition",
       COARSE_SIGMOID_ADDITION,
       {
           {1, 4, 0},
           {2, 4, 0},
           {3, 4, 0},
           {4, 4, 0},
           {5, 4, 0},
           {6, 4, 0},
           {7, 4, 0},
           {8, 4, 0},
       }},
      {2,
       "coarse_tanh_addition",
       COARSE_TANH_ADDITION,
       {
           {1, 4, 0},
           {2, 4, 0},
           {3, 4, 0},
           {4, 4, 0},
           {5, 4, 0},
           {6, 4, 0},
           {7, 4, 0},
           {8, 4, 0},
       }},
      {2,
       "fine_sigmoid_addition",
       FINE_SIGMOID_ADDITION,
       {
           {1, 4, 0},
           {2, 4, 0},
           {3, 4, 0},
           {4, 4, 0},
           {5, 4, 0},
           {6, 4, 0},
           {7, 4, 0},
           {8, 4, 0},
       }},
      {2,
       "fine_tanh_addition",
       FINE_TANH_ADDITION,
       {
           {1, 4, 0},
           {2, 4, 0},
           {3, 4, 0},
           {4, 4, 0},
           {5, 4, 0},
           {6, 4, 0},
           {7, 4, 0},
           {8, 4, 0},
       }},
      {2,
       "hidden",
       HIDDEN,
       {
           {1, 4, 0},
           {2, 4, 0},
           {3, 4, 0},
           {4, 4, 0},
           {5, 4, 0},
           {6, 4, 0},
           {7, 4, 0},
           {8, 4, 0},
       }},
  };

  // open all files
  ofstream srcfile;
  srcfile.open("skylark/inference/blas/ActivationsFP32UKernels.cc");
  srcfile << "/*\n"
             " * Copyright (c) LAIX, Inc. and its affiliates.\n"
             " * All rights reserved.\n"
             " * This source code is licensed under the BSD-style license "
             "found in the\n"
             " * LICENSE file in the root directory of this source tree.\n"
             " */\n";
  srcfile
      << "#include \"skylark/inference/blas/ActivationsFP32UKernels.h\"\n\n";
  srcfile << "namespace fbgemm {\n\n";
  if (iaca) {
    srcfile << "#include \"iacaMarks.h\"\n";
  }

  ofstream hdrfile;
  hdrfile.open("skylark/inference/blas/ActivationsFP32UKernels.h");
  hdrfile << "/*\n"
             " * Copyright (c) LAIX, Inc. and its affiliates.\n"
             " * All rights reserved.\n"
             " * This source code is licensed under the BSD-style license "
             "found in the\n"
             " * LICENSE file in the root directory of this source tree.\n"
             " */\n";

  hdrfile << "#ifndef LAIX_ACTIVATIONS_UKERNELS\n";
  hdrfile << "#define LAIX_ACTIVATIONS_UKERNELS\n\n";
  hdrfile << "#include <cstdint>\n";
  hdrfile << "#include <tuple>\n";
  hdrfile << "#include <vector>\n";
  hdrfile << "#include \"skylark/inference/blas/ActivationsFP32.h\"\n\n";
  hdrfile << "#include \"skylark/inference/blas/Types.h\"\n\n";
  hdrfile << "namespace fbgemm {\n\n";

  std::map<string, string> fptr_typedef;
  fptr_typedef["fp32"] = "";

  for (auto s : isa) {
    vector<vector<unsigned>> &ukernel_shape = s.shapes;

    vector<string> funcname(ukernel_shape.size()),
        fheader(ukernel_shape.size());
    string fargs;

    string prefix = s.name + "_";
    cout << "Generating code for " << s.name << " " << s.name << "\n";
    if (s.activation == 3) {
      srcfile << "\nnamespace wavernn {\n\n";
      hdrfile << "namespace wavernn {\n\n";
    }

    for (unsigned k = 0; k < ukernel_shape.size(); k++) {
      printf("shape: %d x %d * 32\n", ukernel_shape[k][0], ukernel_shape[k][1]);

      funcname[k] = "wavernnkernel_avx256_" + prefix +
                    to_string(ukernel_shape[k][0]) + "x" +
                    to_string(ukernel_shape[k][1]);

      if (s.activation == RELU || s.activation == SIGMOID ||
          s.activation == TANH) {
        funcname[k] = "activationkernel_avx256_" + prefix +
                      to_string(ukernel_shape[k][0]) + "x" +
                      to_string(ukernel_shape[k][1]);
      }

      string p1 = "ReLUParams* gp";
      if (s.activation == COARSE_SIGMOID_ADDITION) {
        p1 = "CoarseSigmoidParams* gp";
      } else if (s.activation == COARSE_TANH_ADDITION) {
        p1 = "CoarseTanhParams* gp";
      } else if (s.activation == FINE_SIGMOID_ADDITION) {
        p1 = "FineSigmoidParams* gp";
      } else if (s.activation == FINE_TANH_ADDITION) {
        p1 = "FineTanhParams* gp";
      } else if (s.activation == HIDDEN) {
        p1 = "HiddenParams* gp";
      } else if (s.activation == TANH) {
        p1 = "TanhParams* gp";
        if (k == 0) {
          srcfile << "// tanh is modified from "
                     "https://github.com/Microsoft/onnxruntime/commit/"
                     "47551da99451a1f20b34bf83e06c6ccbd638fe13"
                  << "\n";
        }
      } else if (s.activation == SIGMOID) {
        p1 = "SigmoidParams* gp";
        if (k == 0) {
          srcfile << "// sigmoid is modified from "
                     "https://github.com/Microsoft/onnxruntime/commit/"
                     "47551da99451a1f20b34bf83e06c6ccbd638fe13"
                  << "\n";
        }
      } else {
        CHECK(s.activation == RELU);
      }

      fargs = "(" + p1 + ")";

      fheader[k] = "void __attribute__((noinline)) " + funcname[k] + fargs;
      if (k > 0)
        srcfile << "\n";
      srcfile << fheader[k] << " {\n";

      unsigned last_free_ymmreg = 0;
      // produce register block of C
      vector<string> vCtile;

      for (auto c = 0; c < ukernel_shape[k][1]; c++) {
        vCtile.push_back("ymm" + to_string(last_free_ymmreg++));
      }
      CHECK(last_free_ymmreg <= 13);

      srcfile << "  asm volatile(\n";
      srcfile << "#if !defined(__clang__)"
              << "\n";
      addi(srcfile, "mov r14, %[gp]");
      srcfile << "#else\n";
      addi(srcfile, "mov %[gp], %%r14");
      addi(srcfile, ".intel_syntax noprefix");
      srcfile << "#endif\n";

      if (s.activation == RELU) {
        srcfile << "\n      // Copy parameters\n";
        srcfile << "      // m\n";
        addi(srcfile, "mov r8, [r14 + 0]");
        srcfile << "      // b_block_cols\n";
        addi(srcfile, "mov r9, [r14 + 8]");
        srcfile << "      // Z\n";
        addi(srcfile, "mov r10, [r14 + 16]");
        srcfile << "      // ldz\n";
        addi(srcfile, "mov r11, [r14 + 24]");
        srcfile << "      // A\n";
        addi(srcfile, "mov r12, [r14 + 32]");
        srcfile << "      // lda\n";
        addi(srcfile, "mov r13, [r14 + 40]");
        srcfile << "      // Make copies of Z and A\n";
        addi(srcfile, "mov rdx, r10");
        addi(srcfile, "mov rax, r12");
        srcfile << "\n";

        addi(srcfile, "vxorps ymm15,ymm15,ymm15");

        string exitlabel = "L_exit%=";

        addi(srcfile, "mov r14, 0");
        addi(srcfile, "cmp r14, r9");
        addi(srcfile, "jge L_exit%=");
        string label = "loop_inner%=";
        addi(srcfile, label + ":");

        srcfile << "\n";

        int batch_size = ukernel_shape[k][0];
        int unroll_factor = ukernel_shape[k][1];
        for (auto r = 0; r < batch_size; r++) {
          int shift = 0;
          for (auto c = 0; c < vCtile.size(); c++) {
            addi(srcfile, "vmovups " + vCtile[c] + ",YMMWORD PTR [r12 + " +
                              to_string(32 * shift++) + "]");
            addi(srcfile, "vmaxps " + vCtile[c] + ",ymm15," + vCtile[c]);
            addi(srcfile, "vmovups YMMWORD PTR [r10 + " + to_string(32 * c) +
                              "], " + vCtile[c]);
          }
          if (batch_size > 1 && r != (batch_size - 1)) {
            addi(srcfile, "add r12, r13"); // move A ptr
            addi(srcfile, "add r10, r11"); // move D ptr
          }
          srcfile << "\n";
        }

        if (batch_size > 1) {
          addi(srcfile, "add rax, " + to_string(32 * unroll_factor));
          addi(srcfile, "add rdx, " + to_string(32 * unroll_factor));
          addi(srcfile, "mov r12, rax");
          addi(srcfile, "mov r10, rdx");
        } else {
          addi(srcfile, "add r12, " + to_string(32 * unroll_factor));
          addi(srcfile, "add r10, " + to_string(32 * unroll_factor));
        }

        srcfile << "\n";
        addi(srcfile, "add r14, " + to_string(unroll_factor));
        addi(srcfile, "cmp r14, r9");
        addi(srcfile, "jge " + exitlabel);
        addi(srcfile, "jmp " + label);

        srcfile << "\n";
        addi(srcfile, exitlabel + ":");
        // output
        srcfile << "      :\n";
        // input
        srcfile << "      : [gp] \"rm\"(gp)\n";
        // clobbered
        srcfile << "      : \"r8\",\n        \"r9\",\n        \"r10\",\n"
                   "        \"r11\",\n        \"r12\",\n        \"r13\",\n"
                   "        \"r14\",\n        \"rax\",\n        \"rdx\",\n"
                   "        \"memory\");\n";
        srcfile << "}\n";
      } else if (s.activation == TANH || s.activation == SIGMOID) {
        CHECK(last_free_ymmreg <= 4);
        srcfile << "\n      // Copy parameters\n";
        srcfile << "      // m\n";
        addi(srcfile, "mov r8, [r14 + 0]");
        srcfile << "      // b_block_cols\n";
        addi(srcfile, "mov r9, [r14 + 8]");
        srcfile << "      // Z\n";
        addi(srcfile, "mov r10, [r14 + 16]");
        srcfile << "      // ldz\n";
        addi(srcfile, "mov r11, [r14 + 24]");
        srcfile << "      // A\n";
        addi(srcfile, "mov r12, [r14 + 32]");
        srcfile << "      // lda\n";
        addi(srcfile, "mov r13, [r14 + 40]");
        if (s.activation == TANH) {
          srcfile << "      // TanhConstants\n";
        } else {
          CHECK(s.activation == SIGMOID);
          srcfile << "      // SigmoidConstants\n";
        }
        addi(srcfile, "mov rcx, [r14 + 48]");
        srcfile << "      // Make copies of Z and A\n";
        addi(srcfile, "mov rdx, r10");
        addi(srcfile, "mov rax, r12");
        srcfile << "\n";

        if (s.activation == TANH) {
          addi(srcfile, "vbroadcastss ymm4,DWORD PTR [rcx + 0]");
          addi(srcfile, "vbroadcastss ymm5,DWORD PTR [rcx + 4]");
          addi(srcfile, "vbroadcastss ymm6,DWORD PTR [rcx + 8]");
          addi(srcfile, "vbroadcastss ymm7,DWORD PTR [rcx + 12]");
          addi(srcfile, "vbroadcastss ymm8,DWORD PTR [rcx + 16]");
          addi(srcfile, "vbroadcastss ymm9,DWORD PTR [rcx + 20]");
          addi(srcfile, "vbroadcastss ymm10,DWORD PTR [rcx + 24]");
          addi(srcfile, "vbroadcastss ymm11,DWORD PTR [rcx + 28]");
          addi(srcfile, "vbroadcastss ymm12,DWORD PTR [rcx + 32]");
          addi(srcfile, "vbroadcastss ymm13,DWORD PTR [rcx + 36]");

          addi(srcfile, "vbroadcastss ymm14,DWORD PTR [rcx + 44]");
          addi(srcfile, "vbroadcastss ymm15,DWORD PTR [rcx + 48]");
        } else {
          addi(srcfile, "vbroadcastss ymm4,DWORD PTR [rcx + 0]");
          addi(srcfile, "vbroadcastss ymm5,DWORD PTR [rcx + 4]");
          addi(srcfile, "vbroadcastss ymm6,DWORD PTR [rcx + 8]");
          addi(srcfile, "vbroadcastss ymm7,DWORD PTR [rcx + 12]");
          addi(srcfile, "vbroadcastss ymm8,DWORD PTR [rcx + 16]");
          addi(srcfile, "vbroadcastss ymm9,DWORD PTR [rcx + 20]");
          addi(srcfile, "vbroadcastss ymm10,DWORD PTR [rcx + 24]");
          addi(srcfile, "vbroadcastss ymm11,DWORD PTR [rcx + 28]");

          addi(srcfile, "vbroadcastss ymm12,DWORD PTR [rcx + 36]");
          addi(srcfile, "vbroadcastss ymm13,DWORD PTR [rcx + 40]");
          addi(srcfile, "vbroadcastss ymm14,DWORD PTR [rcx + 44]");
          addi(srcfile, "vbroadcastss ymm15,DWORD PTR [rcx + 48]");
        }
        srcfile << "\n";

        string exitlabel = "L_exit%=";

        addi(srcfile, "mov r14, 0");
        addi(srcfile, "cmp r14, r9");
        addi(srcfile, "jge L_exit%=");
        string label = "loop_inner%=";
        addi(srcfile, label + ":");

        srcfile << "\n";

        int batch_size = ukernel_shape[k][0];
        int unroll_factor = ukernel_shape[k][1];
        // clang-format off
        for (auto r = 0; r < batch_size; r++) {
          int shift = 0;
          for (auto c = 0; c < vCtile.size(); c++) {
            if (s.activation == TANH) {
              addi(srcfile, "vmaxps " + vCtile[c] + ",ymm4,YMMWORD PTR [r12 + " + to_string(32 * shift++) + "]   # clamp lower bound");
              addi(srcfile, "vmovaps ymm2,ymm7");
              addi(srcfile, "vminps  ymm0,ymm5,ymm0                  # clamp upper bound");
              addi(srcfile, "vmulps  ymm1,ymm0,ymm0                  # x2");
              addi(srcfile, "vbroadcastss ymm3,DWORD PTR [rcx + 40]");
              addi(srcfile, "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11");
              addi(srcfile, "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9");
              addi(srcfile, "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7");
              addi(srcfile, "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5");
              addi(srcfile, "vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3");
              addi(srcfile, "vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1");
              addi(srcfile, "vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4");
              addi(srcfile, "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2");
              addi(srcfile, "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0");
              addi(srcfile, "vmulps  ymm2,ymm0,ymm2                  # p = x * p");
              addi(srcfile, "vdivps  ymm0,ymm2,ymm3                  # tanh = p / q");
              addi(srcfile, "vmovups YMMWORD PTR [r10 + " + to_string(32 * c) + "], " + vCtile[c]);
              CHECK(vCtile[c] == "ymm0");
            } else {
              // addi(srcfile, "vmaxps  ymm0,ymm4,YMMWORD PTR [rdi]     # clamp lower bound");
              addi(srcfile, "vmaxps " + vCtile[c] + ",ymm4,YMMWORD PTR [r12 + " + to_string(32 * shift++) + "]   # clamp lower bound");
              addi(srcfile, "vmovaps ymm2,ymm7");
              addi(srcfile, "vminps  ymm0,ymm5,ymm0                  # clamp upper bound");
              addi(srcfile, "vmulps  ymm1,ymm0,ymm0                  # x2");
              addi(srcfile, "vbroadcastss ymm3,DWORD PTR [rcx + 32]");
              addi(srcfile, "vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7");
              addi(srcfile, "vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5");
              addi(srcfile, "vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3");
              addi(srcfile, "vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1");
              addi(srcfile, "vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8");
              addi(srcfile, "vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6");
              addi(srcfile, "vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4");
              addi(srcfile, "vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2");
              addi(srcfile, "vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0");
              addi(srcfile, "vmulps  ymm2,ymm0,ymm2                  # p = x * p");
              addi(srcfile, "vbroadcastss ymm0,DWORD PTR [rcx + 52]");
              addi(srcfile, "vdivps  ymm2,ymm2,ymm3");
              addi(srcfile, "vxorps  ymm3,ymm3,ymm3");
              addi(srcfile, "vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5");
              addi(srcfile, "vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound");
              addi(srcfile, "vmovups YMMWORD PTR [r10 + " + to_string(32 * c) + "], " + vCtile[c]);
              CHECK(vCtile[c] == "ymm0");
            }
          }
          if (batch_size > 1 && r != (batch_size - 1)) {
            addi(srcfile, "add r12, r13");  // move A ptr
            addi(srcfile, "add r10, r11");  // move D ptr
          }
          srcfile << "\n";
        }
        // clang-format on

        if (batch_size > 1) {
          addi(srcfile, "add rax, " + to_string(32 * unroll_factor));
          addi(srcfile, "add rdx, " + to_string(32 * unroll_factor));
          addi(srcfile, "mov r12, rax");
          addi(srcfile, "mov r10, rdx");
        } else {
          addi(srcfile, "add r12, " + to_string(32 * unroll_factor));
          addi(srcfile, "add r10, " + to_string(32 * unroll_factor));
        }

        srcfile << "\n";
        addi(srcfile, "add r14, " + to_string(unroll_factor));
        addi(srcfile, "cmp r14, r9");
        addi(srcfile, "jge " + exitlabel);
        addi(srcfile, "jmp " + label);

        srcfile << "\n";
        addi(srcfile, exitlabel + ":");
        // output
        srcfile << "      :\n";
        // input
        srcfile << "      : [gp] \"rm\"(gp)\n";
        // clobbered
        srcfile << "      : \"r8\",\n        \"r9\",\n        \"r10\",\n"
                   "        \"r11\",\n        \"r12\",\n        \"r13\",\n"
                   "        \"r14\",\n        \"rax\",\n        \"rdx\",\n"
                   "        \"memory\");\n";
        srcfile << "}\n";
      } else if (s.activation == COARSE_TANH_ADDITION ||
                 s.activation == COARSE_SIGMOID_ADDITION ||
                 s.activation == FINE_SIGMOID_ADDITION ||
                 s.activation == FINE_TANH_ADDITION || s.activation == HIDDEN) {
        srcfile << "\n      // Copy parameters\n";
        srcfile << "      // b_block_cols\n";
        addi(srcfile, "mov r8, [r14 + 8]");
        srcfile << "      // Z\n";
        addi(srcfile, "mov rsi, [r14 + 16]");
        srcfile << "      // ldz\n";
        addi(srcfile, "mov r9, [r14 + 24]");
        if (s.activation == HIDDEN) {
          srcfile << "      // A\n";
          addi(srcfile, "mov rax, [r14 + 32]");
          srcfile << "      // lda\n";
          addi(srcfile, "mov r11, [r14 + 40]");
          srcfile << "      // B\n";
          addi(srcfile, "mov rbx, [r14 + 48]");
          srcfile << "      // ldb\n";
          addi(srcfile, "mov r12, [r14 + 56]");
          srcfile << "      // C\n";
          addi(srcfile, "mov rcx, [r14 + 64]");
          srcfile << "      // ldc\n";
          addi(srcfile, "mov r10, [r14 + 72]");
        } else {
          srcfile << "      // C\n";
          addi(srcfile, "mov rcx, [r14 + 32]");
          srcfile << "      // ldc\n";
          addi(srcfile, "mov r10, [r14 + 40]");
          srcfile << "      // A\n";
          addi(srcfile, "mov rax, [r14 + 48]");
          srcfile << "      // lda\n";
          addi(srcfile, "mov r11, [r14 + 56]");
          srcfile << "      // B\n";
          addi(srcfile, "mov rbx, [r14 + 64]");
          srcfile << "      // ldb\n";
          addi(srcfile, "mov r12, [r14 + 72]");
        }
        if (s.activation == COARSE_TANH_ADDITION ||
            s.activation == FINE_SIGMOID_ADDITION ||
            s.activation == FINE_TANH_ADDITION) {
          srcfile << "      // D\n";
          addi(srcfile, "mov rdx, [r14 + 80]");
          srcfile << "      // ldd\n";
          addi(srcfile, "mov r13, [r14 + 88]");

          if (s.activation == FINE_TANH_ADDITION) {
            srcfile << "      // T\n";
            addi(srcfile, "mov rdi, [r14 + 96]");
            srcfile << "      // ldt\n";
            addi(srcfile, "mov rbp, [r14 + 104]");
          }
        }

        srcfile << "\n";
        string exitlabel = "L_exit%=";

        addi(srcfile, "mov r14, 0");
        addi(srcfile, "cmp r14, r8");
        addi(srcfile, "jge L_exit%=");
        string label = "loop_inner%=";
        addi(srcfile, label + ":");

        srcfile << "\n";

        int batch_size = ukernel_shape[k][0];
        int unroll_factor = ukernel_shape[k][1];
        // clang-format off
        for (auto r = 0; r < batch_size; r++) {
          int shift = 0;
          for (auto c = 0; c < vCtile.size(); c++) {
            if (s.activation == COARSE_TANH_ADDITION) {
              addi(srcfile, "vmovups " + vCtile[c] + ",YMMWORD PTR [rcx + " + to_string(32 * shift) + "]");
              addi(srcfile, "vmovups ymm13,YMMWORD PTR [rdx + " + to_string(32 * shift) + "]");
              addi(srcfile, "vaddps " + vCtile[c] + "," + vCtile[c] + ",ymm13");
              addi(srcfile, "vmovups ymm15,YMMWORD PTR [rax + " + to_string(32 * shift) + "]");
              addi(srcfile, "vmovups ymm14,YMMWORD PTR [rbx + " + to_string(32 * shift) + "]");
              addi(srcfile, "vfmadd231ps " + vCtile[c] + ",ymm15,ymm14");
              addi(srcfile, "vmovups YMMWORD PTR [rsi + " + to_string(32 * c) + "], " + vCtile[c]);
            } else if (s.activation == COARSE_SIGMOID_ADDITION) {
              addi(srcfile, "vmovups ymm15,YMMWORD PTR [rax + " + to_string(32 * shift) + "]");
              addi(srcfile, "vmovups ymm14,YMMWORD PTR [rbx + " + to_string(32 * shift) + "]");
              addi(srcfile, "vaddps ymm13,ymm15,ymm14");
              addi(srcfile, "vmovups " + vCtile[c] + ",YMMWORD PTR [rcx + " + to_string(32 * shift) + "]");
              addi(srcfile, "vaddps " + vCtile[c] + "," + vCtile[c] + ",ymm13");
              addi(srcfile, "vmovups YMMWORD PTR [rsi + " + to_string(32 * c) + "], " + vCtile[c]);
            } else if (s.activation == HIDDEN) {
              // a
              addi(srcfile, "vmovups ymm15,YMMWORD PTR [rax + " + to_string(32 * shift) + "]");
              // b
              addi(srcfile, "vmovups ymm14,YMMWORD PTR [rbx + " + to_string(32 * shift) + "]");
              // c
              addi(srcfile, "vmovups " + vCtile[c] + ",YMMWORD PTR [rcx + " + to_string(32 * shift) + "]");
              // b - c
              addi(srcfile, "vsubps ymm13,ymm14," + vCtile[c]);
              // z = a * (b - c) + c
              addi(srcfile, "vfmadd231ps " + vCtile[c] + ",ymm15,ymm13");
              addi(srcfile, "vmovups YMMWORD PTR [rsi + " + to_string(32 * c) + "], " + vCtile[c]);
            } else if (s.activation == FINE_SIGMOID_ADDITION) {
              addi(srcfile, "vmovups ymm15,YMMWORD PTR [rax + " +  to_string(32 * shift) + "]");
              addi(srcfile, "vmovups ymm14,YMMWORD PTR [rbx + " + to_string(32 * shift) + "]");
              addi(srcfile, "vaddps ymm13,ymm15,ymm14");
              addi(srcfile, "vmovups ymm15,YMMWORD PTR [rcx + " + to_string(32 * shift) + "]");
              addi(srcfile, "vmovups ymm14,YMMWORD PTR [rdx + " + to_string(32 * shift) + "]");
              addi(srcfile, "vaddps " + vCtile[c] + ",ymm15,ymm14");
              addi(srcfile, "vaddps " + vCtile[c] + "," + vCtile[c] + ",ymm13");
              addi(srcfile, "vmovups YMMWORD PTR [rsi + " + to_string(32 * c) + "], " + vCtile[c]);
            } else if (s.activation == FINE_TANH_ADDITION) {
              addi(srcfile, "vmovups " + vCtile[c] + ",YMMWORD PTR [rcx + " + to_string(32 * shift) + "]");
              addi(srcfile, "vmovups ymm13,YMMWORD PTR [rdx + " + to_string(32 * shift) + "]");
              addi(srcfile, "vaddps " + vCtile[c] + "," + vCtile[c] + ",ymm13");
              addi(srcfile, "vmovups ymm13,YMMWORD PTR [rdi + " + to_string(32 * shift) + "]");
              addi(srcfile, "vaddps " + vCtile[c] + "," + vCtile[c] + ",ymm13");
              addi(srcfile, "vmovups ymm15,YMMWORD PTR [rax + " + to_string(32 * shift) + "]");
              addi(srcfile, "vmovups ymm14,YMMWORD PTR [rbx + " + to_string(32 * shift) + "]");
              addi(srcfile, "vfmadd231ps " + vCtile[c] + ",ymm15,ymm14");
              addi(srcfile, "vmovups YMMWORD PTR [rsi + " + to_string(32 * c) + "], " + vCtile[c]);
            } else {
              CHECK(0);
            }
            srcfile << "\n";
            shift++;
          }
          if (batch_size > 1 && r != (batch_size - 1)) {
            addi(srcfile, "add rsi, r9");   // move Z ptr
            addi(srcfile, "add rcx, r10");  // move C ptr
            addi(srcfile, "add rax, r11");  // move A ptr
            addi(srcfile, "add rbx, r12");  // move B ptr
            if (s.activation == COARSE_TANH_ADDITION ||
                s.activation == FINE_SIGMOID_ADDITION ||
                s.activation == FINE_TANH_ADDITION) {
              addi(srcfile, "add rdx, r13");  // move D ptr
              if (s.activation == FINE_TANH_ADDITION) {
                addi(srcfile, "add rdi, rbp");
              }
            }
          }
          // srcfile << "\n";
        }
        // clang-format on

        if (batch_size == 2) {
          addi(srcfile, "sub rsi, r9");
          addi(srcfile, "sub rcx, r10");
          addi(srcfile, "sub rax, r11");
          addi(srcfile, "sub rbx, r12");
          if (s.activation == COARSE_TANH_ADDITION ||
              s.activation == FINE_SIGMOID_ADDITION ||
              s.activation == FINE_TANH_ADDITION) {
            addi(srcfile, "sub rdx, r13");
            if (s.activation == FINE_TANH_ADDITION) {
              addi(srcfile, "sub rdi, rbp");
            }
          }
          srcfile << "\n";
        } else if (batch_size > 2) {
          addi(srcfile, "imul r15, r9, " + to_string(batch_size - 1));
          addi(srcfile, "sub rsi, r15");
          addi(srcfile, "imul r15, r10, " + to_string(batch_size - 1));
          addi(srcfile, "sub rcx, r15");
          addi(srcfile, "imul r15, r11, " + to_string(batch_size - 1));
          addi(srcfile, "sub rax, r15");
          addi(srcfile, "imul r15, r12, " + to_string(batch_size - 1));
          addi(srcfile, "sub rbx, r15");
          if (s.activation == COARSE_TANH_ADDITION ||
              s.activation == FINE_SIGMOID_ADDITION ||
              s.activation == FINE_TANH_ADDITION) {
            addi(srcfile, "imul r15, r13, " + to_string(batch_size - 1));
            addi(srcfile, "sub rdx, r15");
            if (s.activation == FINE_TANH_ADDITION) {
              addi(srcfile, "imul r15, rbp, " + to_string(batch_size - 1));
              addi(srcfile, "sub rdi, r15");
            }
          }
          srcfile << "\n";
        }
        addi(srcfile, "add rsi, " + to_string(32 * unroll_factor));
        addi(srcfile, "add rax, " + to_string(32 * unroll_factor));
        addi(srcfile, "add rbx, " + to_string(32 * unroll_factor));
        addi(srcfile, "add rcx, " + to_string(32 * unroll_factor));
        if (s.activation == COARSE_TANH_ADDITION ||
            s.activation == FINE_SIGMOID_ADDITION ||
            s.activation == FINE_TANH_ADDITION) {
          addi(srcfile, "add rdx, " + to_string(32 * unroll_factor));
          if (s.activation == FINE_TANH_ADDITION) {
            addi(srcfile, "add rdi, " + to_string(32 * unroll_factor));
          }
        }

        srcfile << "\n";
        addi(srcfile, "add r14, " + to_string(unroll_factor));
        addi(srcfile, "cmp r14, r8");
        addi(srcfile, "jge " + exitlabel);
        addi(srcfile, "jmp " + label);

        srcfile << "\n";
        addi(srcfile, exitlabel + ":");
        // output
        srcfile << "      :\n";
        // input
        srcfile << "      : [gp] \"rm\"(gp)\n";
        // clobbered
        srcfile << "      : \"r8\",\n        \"r9\",\n        \"r10\",\n"
                   "        \"r11\",\n        \"r12\",\n        \"r13\",\n"
                   "        \"r14\",\n        \"r15\",\n        \"rax\",\n"
                   "        \"rbx\",\n        \"rcx\",\n        \"rdx\",\n"
                   "        \"rsi\",\n        \"rdi\",\n"
                   "        \"memory\");\n";
        srcfile << "}\n";

      } else {
        CHECK(0);
      }
    }

    for (unsigned k = 0; k < ukernel_shape.size(); k++) {
      printf("k = %d - %s\n", k, fheader[k].c_str());
      hdrfile << fheader[k] << ";\n";
    }

    // hdrfile << "typedef void (*funcptr_" + s.name + ")" + fargs << ";\n\n";
    hdrfile << "\n\n";
  }
  srcfile << "\n}  // namespace wavernn\n";
  srcfile << "\n}  // namespace fbgemm\n";
  srcfile.close();

  hdrfile << "}  // namespace wavernn\n";
  hdrfile << "\n}  // namespace fbgemm\n\n";
  hdrfile << "#endif  // LAIX_ACTIVATIONS_UKERNELS\n";
  hdrfile.close();
}
