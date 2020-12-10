/* Copyright 2018 LingoChamp Inc. All rights reserved.

 * INTERNAL USE ONLY. DO NOT DISTRIBUTE!

 * Author: Algorithm@liulishuo.com
==============================================================================*/

// A program with a main that is suitable for unittests, including those
// that also define microbenchmarks.  Based on whether the user specified
// the --benchmark_filter flag which specifies which benchmarks to run,
// we will either run benchmarks or run the gtest tests in the program.

#if defined(PLATFORM_GOOGLE) || defined(__ANDROID__)

// main() is supplied by gunit_main
#else
#include "gtest/gtest.h"

GTEST_API_ int main(int argc, char **argv) {
  std::cout << "Running main() from test_main.cc\n";

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
