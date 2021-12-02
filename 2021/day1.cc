#include <benchmark/benchmark.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <immintrin.h>
#include <memory>
#include <vector>

/*
    Run on (1 X 2494.11 MHz CPU )
    CPU Caches:
    L1 Data 32 KiB (x1)
    L1 Instruction 32 KiB (x1)
    L2 Unified 4096 KiB (x1)
    Load Average: 0.44, 0.17, 0.07
    -----------------------------------------------------------
    Benchmark                 Time             CPU   Iterations
    -----------------------------------------------------------
    BM_avx                  133 ns          132 ns      4963443
    BM_manual_stride       1270 ns         1266 ns       532658
    BM_compiler             295 ns          293 ns      2411555
*/

static void escape(void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

std::vector<unsigned short> load_input(const std::string &input_path) {
  std::vector<unsigned short> result;
  std::ifstream input(input_path);
  std::string line;
  while (std::getline(input, line)) {
    result.emplace_back(std::stoi(line));
  }

  return result;
}

static int avx(const unsigned short* depths, int n) {
    int i = 1;
    __m256i incremental = _mm256_set1_epi16(0);
    __m256i mask = _mm256_set_epi16(
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1
    );
    for (; i < n - 16; i += 16) {
        __m256i previous = _mm256_loadu_si256 ((__m256i*) &depths[i - 1]);
        __m256i current = _mm256_loadu_si256 ((__m256i*) &depths[i]);
        __m256i comparisons = _mm256_cmpgt_epi32(current, previous);
        __m256i masked = _mm256_and_si256(comparisons, mask);
        incremental = _mm256_add_epi32(incremental, masked);
    }
    escape(&incremental);

    // Kinda unfortunate that the AVX function requires a literal.
    int increases = 0;
    increases += _mm256_extract_epi16(incremental, 0);
    increases += _mm256_extract_epi16(incremental, 1);
    increases += _mm256_extract_epi16(incremental, 2);
    increases += _mm256_extract_epi16(incremental, 3);
    increases += _mm256_extract_epi16(incremental, 4);
    increases += _mm256_extract_epi16(incremental, 5);
    increases += _mm256_extract_epi16(incremental, 6);
    increases += _mm256_extract_epi16(incremental, 7);
    increases += _mm256_extract_epi16(incremental, 8);
    increases += _mm256_extract_epi16(incremental, 9);
    increases += _mm256_extract_epi16(incremental, 10);
    increases += _mm256_extract_epi16(incremental, 11);
    increases += _mm256_extract_epi16(incremental, 12);
    increases += _mm256_extract_epi16(incremental, 13);
    increases += _mm256_extract_epi16(incremental, 14);
    increases += _mm256_extract_epi16(incremental, 15);

    for (; i < n; i++) {
        increases += depths[i] > depths[i - 1];
    }

    return increases;
}

static int manual_stride(const unsigned short* depths, int n) {
    int i = 1;
    constexpr int unroll = 2;
    int incrementals[unroll] = {0};
    int x = 0, y = 0;
    int lim = n - unroll;
    for (; i < lim; i += unroll) {
        for (int j = 0; j < unroll; j++) {
            incrementals[j] += depths[i + j] > depths[i + j - 1];
        }
    }

    int increases = 0;
    for (int j = 0; j < unroll; j++) {
        increases += incrementals[j];
    }

    for (; i < n; i++) {
        increases += depths[i] > depths[i - 1];
    }

    return increases;
}

static void BM_manual_stride(benchmark::State& state) {
    const auto depths = load_input("day1.input");
    for (auto _ : state) {
        int increases = manual_stride(depths.data(), depths.size());
        escape(&increases);
    }
}


int compiler(const std::vector<unsigned short> &data) {
  int increases = 0;
  for (std::size_t i = 1; i < data.size(); ++i) {
    increases += data[i] > data[i - 1];
  }

  return increases;
}

static void BM_compiler(benchmark::State &state) {
    const auto depths = load_input("day1.input");
    int result;

    for (auto _ : state) {
        result = compiler(depths);
        escape(&result);
    }
}

static void BM_avx(benchmark::State& state) {
    const auto depths = load_input("day1.input");

    int increases;
    for (auto _ : state) {
        increases = avx(depths.data(), depths.size());
        escape(&increases);
    }
}

BENCHMARK(BM_avx);
BENCHMARK(BM_manual_stride);
BENCHMARK(BM_compiler);

BENCHMARK_MAIN();