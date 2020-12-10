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

struct ISA {
  unsigned avx; // 1, 2 or 3
  string name;
  vector<vector<unsigned>> shapes;
};

int main() {
  bool iaca = false;

  int eax, ebx, ecx, edx;
  __cpuid(1 /* ecx = vendor string */, eax, ebx, ecx, edx);
  printf("FC16 is %s supported\n", ((ecx & bit_F16C) ? " " : "not"));

  string comma = ",";

  const int M = 4;
  const string breg = "ymm15";
  const string r_spare = "ymm14";

  vector<ISA> isa = {
      // {1, "AVX", {{4, 1, 0}, {4, 2, 0}, {4, 3, 0}, {3, 1, 0}, {3, 2, 0}, {3,
      // 3, 0}}},
      {2,
       "AVX256",
       {
           {1, 2, 0},
           {2, 2, 0},
           {3, 2, 0},
           {4, 2, 0},
           {5, 1, 0},
           {6, 1, 0},
           {7, 1, 0},
           {8, 1, 0},
           {9, 1, 0},
           {10, 1, 0},
           {11, 1, 0},
           {12, 1, 0},
           {13, 1, 0},
           {14, 1, 0},
       }}};

  // open all files
  ofstream srcfile;
  srcfile.open("skylark/inference/blas/FbgemmFP16UKernels.cc");
  srcfile << "/*\n"
             " * Copyright (c) LAIX, Inc. and its affiliates.\n"
             " * All rights reserved.\n"
             " * This source code is licensed under the BSD-style license "
             "found in the\n"
             " * LICENSE file in the root directory of this source tree.\n"
             " */\n";
  srcfile << "#include \"skylark/inference/blas/FbgemmFP16UKernels.h\"\n\n";
  srcfile << "namespace fbgemm {\n\n";
  if (iaca) {
    srcfile << "#include \"iacaMarks.h\"\n";
  }

  ofstream hdrfile;
  hdrfile.open("skylark/inference/blas/FbgemmFP16UKernels.h");
  hdrfile << "/*\n"
             " * Copyright (c) LAIX, Inc. and its affiliates.\n"
             " * All rights reserved.\n"
             " * This source code is licensed under the BSD-style license "
             "found in the\n"
             " * LICENSE file in the root directory of this source tree.\n"
             " */\n";

  hdrfile << "#ifndef FBGEMM_UKERNELS\n";
  hdrfile << "#define FBGEMM_UKERNELS\n";
  hdrfile << "#include <cstdint>\n";
  hdrfile << "#include <tuple>\n";
  hdrfile << "#include <vector>\n";
  hdrfile << "#include \"skylark/inference/blas/Types.h\"\n\n";
  hdrfile << "namespace fbgemm {\n\n";

  /*
  hdrfile << "using fp16 = float16;\n";
  hdrfile << "using fp32 = float;\n";
  hdrfile << "struct GemmParams {\n  uint64_t k;\n  const float* A;\n  const "
             "fp16* B;\n"
             "  const float* beta;\n  uint64_t accum;\n  float* C;\n  uint64_t "
             "ldc;\n"
             "  uint64_t b_block_cols;\n  uint64_t b_block_size;\n};\n";
  */

  std::map<string, string> fptr_typedef;
  fptr_typedef["fp16"] = "";
  fptr_typedef["fp32"] = "";

#if 1
  for (auto fixedA : {false})
    for (auto fixedB : {false})
      for (auto fixedC : {false})
#else
  for (auto fixedA : {true})
    for (auto fixedB : {true})
      for (auto fixedC : {true})
#endif
        for (auto s : isa) {
          vector<vector<unsigned>> &ukernel_shape = s.shapes;

          vector<string> funcname(ukernel_shape.size()),
              fheader(ukernel_shape.size());
          string fargs;

          for (auto fp16 : {true}) {
            string B_type = ((fp16) ? "fp16" : "fp32");
            string prefix = s.name + /*"_" + B_type */ +"_" + "fA" +
                            to_string(fixedA) + "fB" + to_string(fixedB) +
                            "fC" + to_string(fixedC);
            cout << "Generating code for " << s.name << " " << B_type << "\n";

            for (unsigned k = 0; k < ukernel_shape.size(); k++) {
              printf("shape: %d x %d * 32\n", ukernel_shape[k][0],
                     ukernel_shape[k][1]);

              string p1 = "GemmParams* gp";

              funcname[k] = "gemmkernel_" + to_string(ukernel_shape[k][0]) +
                            "x" + to_string(ukernel_shape[k][1]) + "_";
              funcname[k] += prefix;

              fargs = "(" + p1 + ")";

              fheader[k] =
                  "void __attribute__((noinline)) " + funcname[k] + fargs;
              if (k > 0)
                srcfile << "\n";
              srcfile << fheader[k] << " {\n";

              unsigned last_free_ymmreg = 0;
              // produce register block of C
              vector<vector<string>> vCtile(ukernel_shape[k][0]);
              for (auto c = 0; c < ukernel_shape[k][1]; c++) {
                for (auto r = 0; r < ukernel_shape[k][0]; r++) {
                  vCtile[r].push_back("ymm" + to_string(last_free_ymmreg));
                  last_free_ymmreg++;
                }
              }
              CHECK(last_free_ymmreg <= 14);

              string vAtmp = "ymm" + to_string(last_free_ymmreg++);
              // produce register block of B col
              CHECK(ukernel_shape[k][1] == 1 || ukernel_shape[k][1] == 2);
              vector<string> vBcol(ukernel_shape[k][1]);

              for (auto c = 0; c < ukernel_shape[k][1]; c++) {
                vBcol[c] = ("ymm" + to_string(last_free_ymmreg));
                last_free_ymmreg++;
              }

              CHECK(last_free_ymmreg <= 16);

              srcfile << "  asm volatile(\n";

              srcfile << "#if !defined(__clang__)"
                      << "\n";
              addi(srcfile, "mov r14, %[gp]");
              srcfile << "#else\n";
              addi(srcfile, "mov %[gp], %%r14");
              addi(srcfile, ".intel_syntax noprefix");
              srcfile << "#endif\n";

              srcfile << "\n      // Copy parameters\n";
              srcfile << "      // k\n";
              addi(srcfile, "mov r8, [r14 + 0]");
              srcfile << "      // A\n";
              addi(srcfile, "mov r9, [r14 + 8]");
              srcfile << "      // B\n";
              addi(srcfile, "mov r10, [r14 + 16]");
              srcfile << "      // beta\n";
              addi(srcfile, "mov r15, [r14 + 24]");
              srcfile << "      // accum\n";
              addi(srcfile, "mov rdx, [r14 + 32]");
              srcfile << "      // C\n";
              addi(srcfile, "mov r12, [r14 + 40]");
              srcfile << "      // ldc\n";
              addi(srcfile, "mov r13, [r14 + 48]");
              srcfile << "      // b_block_cols\n";
              addi(srcfile, "mov rdi, [r14 + 56]");
              srcfile << "      // b_block_size\n";
              addi(srcfile, "mov rsi, [r14 + 64]");
              srcfile << "      // Make copies of A and C\n";
              addi(srcfile, "mov rax, r9");
              addi(srcfile, "mov rcx, r12");
              srcfile << "\n";

              if (ukernel_shape[k][0] <= M) {
                srcfile << "      // beta\n";
                addi(srcfile, "vbroadcastss " + r_spare + ",DWORD PTR [r15]");
                srcfile << "\n";
              }

              addi(srcfile, "mov rbx, 0");

              string exitlabel = "L_exit%=";
              string label2 = "loop_outter%=";
              addi(srcfile, label2 + ":");

              // set all vCtile regs to zeros
              if (ukernel_shape[k][0] <= M) {
                addi(srcfile, "mov r14, 0");
                int num = 0;
                for (auto c = 0; c < ukernel_shape[k][1]; c++) {
                  for (auto r = 0; r < vCtile.size(); r++) {
                    num++;
                    addi(srcfile, "vxorps " + vCtile[r][c] + "," +
                                      vCtile[r][c] + "," + vCtile[r][c]);
                  }
                }
              } else {
                addi(srcfile, "mov r14, 0");
                for (auto r = 0; r < vCtile.size(); r++) {
                  for (auto c = 0; c < vCtile[r].size(); c++) {
                    addi(srcfile, "vxorps " + vCtile[r][c] + "," +
                                      vCtile[r][c] + "," + vCtile[r][c]);
                  }
                }
              }

              // start marker
              if (iaca) {
                addi(srcfile, "mov ebx, 111");
                addi(srcfile, ".byte 0x64, 0x67, 0x90");
              }

              srcfile << "\n";

              if (ukernel_shape[k][0] <= M) {
                // addi(srcfile, "vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]");
                addi(srcfile, "mov r11, 0");
                addi(srcfile, "mov rbp, rsi");
              } else if (ukernel_shape[k][0] <= 13) {
                addi(srcfile, "vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]");
                addi(srcfile, "mov r11, 16");
              } else {
                addi(srcfile, "mov r11, 0");
              }

              srcfile << "\n";
              string label = "loop_inner%=";
              addi(srcfile, label + ":");
              srcfile << "\n";

              if (ukernel_shape[k][0] <= M) {
                // auto find largeest unroll
                int batch_size = ukernel_shape[k][0];
                int unroll_factor = 1;
                while ((unroll_factor * batch_size) <= (14 - batch_size * 2)) {
                  unroll_factor += 1;
                }
                unroll_factor -= 1;
                unroll_factor = int(unroll_factor / 2) * 2;
                if (unroll_factor > 16) {
                  unroll_factor = 16;
                } else if (unroll_factor > 8) {
                  unroll_factor = 8;
                } else if (unroll_factor <= 0) {
                  unroll_factor = 1;
                }
                printf("batch_size = %d, unroll_factor = %d\n", batch_size,
                       unroll_factor);

                // CHECK((14 - batch_size * 2 - unroll_factor * batch_size) >=
                // last_free_ymmreg);
                for (auto ur = 0; ur < (16 / unroll_factor); ur++) {
                  int a_offset = 0;
                  for (auto c = 0; c < 2; c++) {
                    int shift = 0;
                    for (auto u = 0; u < unroll_factor; u++) {
                      if (c == 0) {
                        addi(srcfile, "vcvtph2ps " + breg +
                                          ",XMMWORD PTR [r10 + r11 + " +
                                          to_string(u * 16) + "]");
                      } else {
                        addi(srcfile, "vcvtph2ps " + breg +
                                          ",XMMWORD PTR [r10 + rbp + " +
                                          to_string(u * 16) + "]");
                      }

                      for (auto r = 0; r < vCtile.size(); r++) {
                        auto atmp = "ymm" + to_string(14 - 1 - u - r - shift);
                        if (c == 0) {
                          addi(srcfile, "vbroadcastss " + atmp +
                                            ",DWORD PTR [r9 + " +
                                            to_string(a_offset) + "]");
                          a_offset += 4;
                        }

                        addi(srcfile, "vfmadd231ps " + vCtile[r][c] + "," +
                                          breg + "," + atmp);
                      }
                      shift += (batch_size - 1);
                    }
                    srcfile << "\n";
                  }

                  addi(srcfile, "add r9, " + to_string(a_offset));
                  addi(srcfile, "add r11, " + to_string(unroll_factor * 16));
                  addi(srcfile, "add rbp, " + to_string(unroll_factor * 16));
                  srcfile << "\n";
                }

                addi(srcfile, "add r14, " + to_string(unroll_factor *
                                                      (16 / unroll_factor)));
                addi(srcfile, "cmp r14, r8");
                addi(srcfile, "jge " + exitlabel);
                addi(srcfile, "jmp " + label);

                srcfile << "\n";

                addi(srcfile, exitlabel + ":");
              } else if (ukernel_shape[k][0] <= 13) {
                auto a_offset = 0, unroll_factor = 2;
                for (auto u = 0; u < unroll_factor; u++) {
                  string breg = (u == 0) ? "ymm14" : "ymm15";
                  string breg_rev = (u == 0) ? "ymm15" : "ymm14";

                  addi(srcfile, "vcvtph2ps " + breg +
                                    ",XMMWORD PTR [r10 + r11 + " +
                                    to_string(u * 16) + "]");
                  addi(srcfile, "inc r14");
                  for (auto r = 0; r < vCtile.size(); r++) {
                    addi(srcfile, "vbroadcastss " + vAtmp + ",DWORD PTR [r9+" +
                                      to_string(a_offset) + "]");
                    addi(srcfile, "vfmadd231ps " + vCtile[r][0] + "," +
                                      breg_rev + "," + vAtmp);
                    if (u == 1 && r == vCtile.size() / 2)
                      addi(srcfile, "add r11, 32");
                    a_offset += 4;
                  }
                  if (u < unroll_factor - 1) {
                    addi(srcfile, "cmp r14, r8");
                    addi(srcfile, "jge " + exitlabel);
                  }
                }

                addi(srcfile, "add r9," + to_string(a_offset));
                addi(srcfile, "cmp r14, r8");
                addi(srcfile, "jl " + label);

                srcfile << "\n";

                addi(srcfile, exitlabel + ":");
              } else {
                addi(srcfile,
                     "vcvtph2ps " + vBcol[0] + ",XMMWORD PTR [r10 + r11]");
                for (auto r = 0; r < vCtile.size(); r++) {
                  addi(srcfile, "vbroadcastss " + vAtmp + ",DWORD PTR [r9+" +
                                    to_string(4 * r) + "]");
                  addi(srcfile, "vfmadd231ps " + vCtile[r][0] + "," + vBcol[0] +
                                    "," + vAtmp);
                }

                addi(srcfile, "add r9," + to_string(4 * ukernel_shape[k][0]),
                     fixedA); // move A ptr
                addi(srcfile, "add r11, 16");

                addi(srcfile, "inc r14");
                addi(srcfile, "cmp r14, r8");
                addi(srcfile, "jl " + label);
              }

              if (ukernel_shape[k][0] <= M) {
                for (auto r = 0; r < ukernel_shape[k][1]; r++)
                  addi(srcfile, "add r10, rsi");
                srcfile << "\n";

                // end marker
                if (iaca) {
                  addi(srcfile, "mov ebx, 222");
                  addi(srcfile, ".byte 0x64, 0x67, 0x90");
                }

                addi(srcfile, "cmp rdx, 1");
                addi(srcfile, "je L_accum%=");
                srcfile << "      // Dump C\n";

                for (auto r = 0; r < vCtile.size(); r++) {
                  for (auto c = 0; c < vCtile[r].size(); c++) {
                    addi(srcfile, "vmovups YMMWORD PTR [r12 + " +
                                      to_string(32 * c) + "], " + vCtile[r][c],
                         fixedC);
                  }
                  addi(srcfile, "add r12, r13", fixedC); // move C ptr
                }
                addi(srcfile, "jmp L_done%=");

                srcfile << "\n";
                addi(srcfile, "L_accum%=:");
                srcfile << "      // Dump C with accumulate\n";

                for (auto r = 0; r < vCtile.size(); r++) {
                  for (auto c = 0; c < vCtile[r].size(); c++) {
                    switch (s.avx) {
                    case 1:
                      addi(srcfile,
                           string("vmulps ymm15, ") + r_spare + comma +
                               "YMMWORD PTR [r12 + " + to_string(32 * c) + "]",
                           fixedC);
                      addi(srcfile, "vaddps " + vCtile[r][c] + "," +
                                        vCtile[r][c] + "," + "ymm15",
                           fixedC);
                      break;
                    case 2:
                      addi(srcfile,
                           "vfmadd231ps " + vCtile[r][c] + "," + r_spare + "," +
                               "YMMWORD PTR [r12 + " + to_string(32 * c) + "]",
                           fixedC);
                      break;
                    default:
                      CHECK(0);
                    }
                    addi(srcfile, "vmovups YMMWORD PTR [r12 + " +
                                      to_string(32 * c) + "], " + vCtile[r][c],
                         fixedC);
                  }
                  addi(srcfile, "add r12, r13", fixedC); // move C ptr
                }

                srcfile << "\n";
                addi(srcfile, "L_done%=:");

                srcfile << "\n      // next outer iteration\n";
                // C
                addi(srcfile, "add rcx, " + to_string(32 * ukernel_shape[k][1]),
                     fixedC);
                addi(srcfile, "mov r12, rcx", fixedC);
                // A
                addi(srcfile, "mov r9, rax");

                addi(srcfile, "add rbx, " + to_string(ukernel_shape[k][1]));
                addi(srcfile, "cmp rbx, rdi");
                addi(srcfile, "jl " + label2);
              } else {
                addi(srcfile, "add r10, rsi");
                srcfile << "\n";

                // end marker
                if (iaca) {
                  addi(srcfile, "mov ebx, 222");
                  addi(srcfile, ".byte 0x64, 0x67, 0x90");
                }

                addi(srcfile, "cmp rdx, 1");
                addi(srcfile, "je L_accum%=");
                srcfile << "      // Dump C\n";

                for (auto r = 0; r < vCtile.size(); r++) {
                  for (auto c = 0; c < vCtile[r].size(); c++) {
                    addi(srcfile, "vmovups YMMWORD PTR [r12 + " +
                                      to_string(32 * c) + "], " + vCtile[r][c],
                         fixedC);
                  }
                  addi(srcfile, "add r12, r13", fixedC); // move C ptr
                }
                addi(srcfile, "jmp L_done%=");

                srcfile << "\n";
                addi(srcfile, "L_accum%=:");
                srcfile << "      // Dump C with accumulate\n";

                string r_spare = (s.avx == 1) ? "ymm14" : "ymm15";
                addi(srcfile,
                     "vbroadcastss " + r_spare + string(",DWORD PTR [r15]"),
                     fixedC);
                // store out C
                for (auto r = 0; r < vCtile.size(); r++) {
                  for (auto c = 0; c < vCtile[r].size(); c++) {
                    switch (s.avx) {
                    case 1:
                      addi(srcfile,
                           string("vmulps ymm15, ") + r_spare + comma +
                               "YMMWORD PTR [r12 + " + to_string(32 * c) + "]",
                           fixedC);
                      addi(srcfile, "vaddps " + vCtile[r][c] + "," +
                                        vCtile[r][c] + "," + "ymm15",
                           fixedC);
                      break;
                    case 2:
                      addi(srcfile,
                           "vfmadd231ps " + vCtile[r][c] + "," + r_spare + "," +
                               "YMMWORD PTR [r12 + " + to_string(32 * c) + "]",
                           fixedC);
                      break;
                    default:
                      CHECK(0);
                    }
                    addi(srcfile, "vmovups YMMWORD PTR [r12 + " +
                                      to_string(32 * c) + "], " + vCtile[r][c],
                         fixedC);
                  }
                  addi(srcfile, "add r12, r13", fixedC); // move C ptr
                }

                srcfile << "\n";
                addi(srcfile, "L_done%=:");

                srcfile << "\n      // next outer iteration\n";
                // C
                addi(srcfile, "add rcx, " + to_string(32 * ukernel_shape[k][1]),
                     fixedC);
                addi(srcfile, "mov r12, rcx", fixedC);
                // A
                addi(srcfile, "mov r9, rax");

                addi(srcfile, "inc rbx");
                addi(srcfile, "cmp rbx, rdi");
                addi(srcfile, "jl " + label2);
              }

              // output
              srcfile << "      :\n";
              // input
              srcfile << "      : [gp] \"rm\"(gp)\n";

              // clobbered
              srcfile
                  << "      : \"r8\",\n        \"r9\",\n        \"r10\",\n"
                     "        \"r11\",\n        \"r15\",\n        \"r13\",\n"
                     "        \"r14\",\n        \"rax\",\n        \"rcx\",\n"
                     "        \"rdx\",\n        \"rsi\",\n        \"rdi\",\n"
                     "        \"rbx\",\n        \"r12\",\n"
                     "        \"memory\");\n";
              srcfile << "}\n";
            }

            for (unsigned k = 0; k < ukernel_shape.size(); k++) {
              hdrfile << fheader[k] << ";\n";
            }

            fptr_typedef[B_type] =
                "typedef void (*funcptr_" + B_type + ")" + fargs;
          }
        }

  srcfile << "\n}  // namespace fbgemm\n";
  srcfile.close();

  hdrfile << fptr_typedef["fp16"] << ";\n";
  hdrfile << fptr_typedef["fp32"] << ";\n";
  hdrfile << "\n}  // namespace fbgemm\n\n";
  hdrfile << "#endif  // FBGEMM_UKERNELS\n";
  hdrfile.close();
}
