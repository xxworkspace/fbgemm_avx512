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
  vector<ISA> isa = {
      // {1, "AVX", {{4, 1, 0}, {4, 2, 0}, {4, 3, 0}, {3, 1, 0}, {3, 2, 0}, {3,
      // 3, 0}}},
      {2,
       "AVX512",
       {
           {1, 2, 0},
           {2, 2, 0},
           {3, 2, 0},
           {4, 2, 0},
           {5, 2, 0},
           {6, 2, 0},
           {7, 2, 0},
           {8, 2, 0},
           {9, 2, 0},
           {10, 2, 0},
           {11, 2, 0},
           {12, 2, 0},
           {13, 2, 0},
           {14, 2, 0},
           {1, 4, 0},
           {2, 4, 0},
           {3, 4, 0},
           {4, 4, 0},
           {5, 4, 0},
           {6, 4, 0},
       }}};

  std::ofstream srcfile("skylark/inference/blas/FbgemmFP16UKernelsAvx512.cc",
                        ios::out);
  if (srcfile.fail()) {
    printf("src file open fail!\n");
    exit(0);
  }

  srcfile << "/*\n"
             " * Copyright (c) LAIX, Inc. and its affiliates.\n"
             " * All rights reserved.\n"
             " * This source code is licensed under the BSD-style license "
             "found in the\n"
             " * LICENSE file in the root directory of this source tree.\n"
             " */\n";
  srcfile
      << "#include \"skylark/inference/blas/FbgemmFP16UKernelsAvx512.h\"\n\n";
  srcfile << "namespace fbgemm {\n\n";
  if (iaca) {
    srcfile << "#include \"iacaMarks.h\"\n";
  }

  ofstream hdrfile("skylark/inference/blas/FbgemmFP16UKernelsAvx512.h");
  if (hdrfile.fail()) {
    printf("head file open fail!\n");
    exit(0);
  }
  hdrfile << "/*\n"
             " * Copyright (c) LAIX, Inc. and its affiliates.\n"
             " * All rights reserved.\n"
             " * This source code is licensed under the BSD-style license "
             "found in the\n"
             " * LICENSE file in the root directory of this source tree.\n"
             " */\n";

  hdrfile << "#ifndef FBGEMM_UKERNELS_AVX512\n";
  hdrfile << "#define FBGEMM_UKERNELS_AVX512\n";
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

  bool fixedA = false, fixedB = false, fixedC = false;
  for (auto s : isa) {
    vector<vector<unsigned>> &ukernel_shape = s.shapes;
    vector<string> funcname(ukernel_shape.size()),
        fheader(ukernel_shape.size());
    string fargs;

    for (auto fp16 : {true}) {
      string B_type = ((fp16) ? "fp16" : "fp32");
      string prefix = s.name + /*"_" + B_type */ +"_" + "fA" +
                      to_string(fixedA) + "fB" + to_string(fixedB) + "fC" +
                      to_string(fixedC);
      cout << "Generating code for " << s.name << " " << B_type << "\n";

      for (unsigned k = 0; k < ukernel_shape.size(); k++) {
        printf("shape: %d x %d * 32\n", ukernel_shape[k][0],
               ukernel_shape[k][1]);

        string p1 = "GemmParams* gp";

        funcname[k] = "gemmkernel_" + to_string(ukernel_shape[k][0]) + "x" +
                      to_string(ukernel_shape[k][1]) + "_";
        //              to_string(ukernel_shape[k][1]) + "_";
        funcname[k] += prefix;
        fargs = "(" + p1 + ")";
        fheader[k] = "void __attribute__((noinline)) " + funcname[k] + fargs;

        if (k > 0)
          srcfile << "\n";
        srcfile << fheader[k] << " {\n";

        unsigned last_free_ymmreg = 0;
        // produce register block of C
        vector<vector<string>> vCtile(ukernel_shape[k][0]);
        for (auto c = 0; c < ukernel_shape[k][0]; c++) {
          for (auto r = 0; r < ukernel_shape[k][1]; r++) {
            vCtile[c].push_back("zmm" + to_string(last_free_ymmreg));
            last_free_ymmreg++;
          }
        }
        CHECK(last_free_ymmreg <= 28);

        // produce register block of B col
        CHECK(ukernel_shape[k][1] == 2 || ukernel_shape[k][1] == 4);
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
        // srcfile << "      // Make copies of A and C\n";
        // addi(srcfile, "mov rax, r9");
        // addi(srcfile, "mov rcx, r12");
        srcfile << "\n";
        srcfile << "      // beta\n";
        addi(srcfile, "vbroadcastss zmm31,DWORD PTR [r15]");
        srcfile << "\n";

        srcfile << "      // outter loop counter\n";
        addi(srcfile, "mov r14,0");
        addi(srcfile, "loop_outter%=:");
        // set all vCtile regs to zeros
        for (auto c = 0; c < ukernel_shape[k][0]; c++) {
          for (auto r = 0; r < ukernel_shape[k][1]; r++) {
            addi(srcfile, "vxorps " + vCtile[c][r] + "," + vCtile[c][r] + "," +
                              vCtile[c][r]);
          }
        }
        srcfile << "\n";
        // a
        addi(srcfile, "mov rax,r9");
        srcfile << "\n";
        srcfile << "      // inner loop counter\n";
        addi(srcfile, "mov r15,0");
        addi(srcfile, "loop_inner%=:");
        srcfile << "\n";

        // unroll loop
        srcfile << "      // unroll loop 16 times\n";
#define UNROOL 16
        for (int l = 0; l < UNROOL; ++l) {
          srcfile << "      // loop : " << l << "\n";
          // read b
          if(ukernel_shape[k][1] == 2){
            addi(srcfile, "vcvtph2ps zmm28,YMMWORD PTR [r10]");
            addi(srcfile, "add r10,32");
            addi(srcfile, "vcvtph2ps zmm29,YMMWORD PTR [r10]");
            addi(srcfile, "add r10,32");
          }
          else if(ukernel_shape[k][1] == 4){
            addi(srcfile, "vcvtph2ps zmm26,YMMWORD PTR [r10]");
            addi(srcfile, "vcvtph2ps zmm28,YMMWORD PTR [r10 + rsi]");
            addi(srcfile, "add r10,32");
            addi(srcfile, "vcvtph2ps zmm27,YMMWORD PTR [r10]");
            addi(srcfile, "vcvtph2ps zmm29,YMMWORD PTR [r10 + rsi]");
            addi(srcfile, "add r10,32");
          }

          srcfile << "\n";

          // compute a*b
          for (int i = 0; i < ukernel_shape[k][0]; ++i) {
            // load a
            addi(srcfile, "vbroadcastss zmm30,DWORD PTR [rax + " +
                              to_string(l * ukernel_shape[k][0] * 4 + i * 4) +
                              "]");
            // a*b
            for(int l = 0 ; l < ukernel_shape[k][1] ; ++l){
              if(ukernel_shape[k][1] == 2)
                addi(srcfile, "vfmadd231ps " + vCtile[i][l] + ",zmm30,zmm" + to_string(28 + l));
              else if(ukernel_shape[k][1] == 4)
                addi(srcfile, "vfmadd231ps " + vCtile[i][l] + ",zmm30,zmm" + to_string(26 + l));
            }
            //addi(srcfile, "vfmadd231ps " + vCtile[i][0] + ",zmm30,zmm28");
            //addi(srcfile, "vfmadd231ps " + vCtile[i][1] + ",zmm30,zmm29");
          }
          // addi(srcfile, "add rax," + to_string(ukernel_shape[k][0] * 4));
          srcfile << "\n";
        }
        addi(srcfile, "add rax," + to_string(UNROOL * ukernel_shape[k][0] * 4));
        srcfile << "      // unroll loop finish\n";
        // update
        // addi(srcfile, "inc r15");
        addi(srcfile, "add r15,16");
        addi(srcfile, "cmp r15,r8");
        addi(srcfile, "jge L_exit%=");
        addi(srcfile, "jmp loop_inner%=");
        srcfile << "\n";

        addi(srcfile, "L_exit%=:");
        // addi(srcfile, "add rbp,rsp");
        addi(srcfile, "cmp rdx,1");
        addi(srcfile, "je L_accum%=");
        srcfile << "\n";
        // dump c
        addi(srcfile, "mov r11,0");
        srcfile << "\n";
        for (int i = 0; i < ukernel_shape[k][0]; ++i) {
          for(int l = 0 ; l < ukernel_shape[k][1]; ++l){
            addi(srcfile, "vmovups ZMMWORD PTR [r12 + r11 + " + to_string(64 * l) + "]," + vCtile[i][l]);
          }
          //addi(srcfile, "vmovups ZMMWORD PTR [r12 + r11]," + vCtile[i][0]);
          //addi(srcfile, "vmovups ZMMWORD PTR [r12 + r11 + 64]," + vCtile[i][1]);
          addi(srcfile, "add r11,r13");
        }
        srcfile << "\n";
        addi(srcfile, "add r12," + to_string(ukernel_shape[k][1] * 64));
        //addi(srcfile, "add r12,128");
        srcfile << "\n";
        addi(srcfile, "jmp L_done%=");
        srcfile << "\n";

        addi(srcfile, "L_accum%=:");
        addi(srcfile, "mov r11,0");
        srcfile << "\n";
        for (int i = 0; i < ukernel_shape[k][0]; ++i) {
          for(int l = 0 ; l < ukernel_shape[k][1]; ++l){
            addi(srcfile, "vfmadd231ps " + vCtile[i][l] +
                            ",zmm31,ZMMWORD PTR [r12 + r11 + " + to_string(64 * l) + "]");
            addi(srcfile, "vmovups ZMMWORD PTR [r12 + r11 + " + to_string(64 * l) + "]," + vCtile[i][l]);
          }
          /*
          addi(srcfile, "vfmadd231ps " + vCtile[i][0] +
                            ",zmm31,ZMMWORD PTR [r12 + r11]");
          addi(srcfile, "vmovups ZMMWORD PTR [r12 + r11]," + vCtile[i][0]);
          addi(srcfile, "vfmadd231ps " + vCtile[i][1] +
                            ",zmm31,ZMMWORD PTR [r12 + r11 + 64]");
          addi(srcfile, "vmovups ZMMWORD PTR [r12 + r11 + 64]," + vCtile[i][1]);
          */
          addi(srcfile, "add r11,r13");
        }
        srcfile << "\n";
        addi(srcfile, "add r12," + to_string(ukernel_shape[k][1] * 64));
        //addi(srcfile, "add r12,128");
        srcfile << "\n";
        addi(srcfile, "L_done%=:");
        // b next
        if(ukernel_shape[k][1] == 4){
          addi(srcfile, "add r10,rsi");
          addi(srcfile, "add r14,2");
        }else
          addi(srcfile, "inc r14");
        // next outer iteration
        addi(srcfile, "cmp r14,rdi");
        addi(srcfile, "jl loop_outter%=");
        srcfile << "\n";
        // output
        srcfile << "      :\n";
        // input
        srcfile << "      : [gp] \"rm\"(gp)\n";
        // clobbered
        srcfile << "      : \"r8\",\n         \"r9\",\n         \"r10\",\n"
                   "        \"r11\",\n        \"r12\",\n        \"r13\",\n"
                   "        \"r14\",\n        \"r15\",\n"
                   "        \"rax\",\n        \"rbx\",\n        \"rcx\",\n"
                   "        \"rdx\",\n        \"rsi\",\n        \"rdi\",\n"
                   "        \"memory\");\n";
        srcfile << "    }\n";

        // for (unsigned k = 0; k < ukernel_shape.size(); k++) {
        hdrfile << fheader[k] << ";\n";
        //}

        fptr_typedef[B_type] = "typedef void (*funcptr_" + B_type + ")" + fargs;
      }
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
