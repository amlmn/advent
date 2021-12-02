#include <benchmark/benchmark.h>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <immintrin.h>
#include <memory>
#include <sstream>
#include <vector>

/*
    Run on (1 X 2494.11 MHz CPU )
    CPU Caches:
      L1 Data 32 KiB (x1)
        L1 Instruction 32 KiB (x1)
          L2 Unified 4096 KiB (x1)
          Load Average: 0.03, 0.01, 0.00
          -----------------------------------------------------------
          Benchmark                 Time             CPU   Iterations
          -----------------------------------------------------------
          BM_avx                  102 ns          102 ns      7633276
          BM_manual_stride       1209 ns         1208 ns       635034
          BM_compiler             283 ns          283 ns      2142797
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

static int sum_to_int(const __m256i& m) {
    int increases = 0;
    increases += _mm256_extract_epi16(m, 0);
    increases += _mm256_extract_epi16(m, 1);
    increases += _mm256_extract_epi16(m, 2);
    increases += _mm256_extract_epi16(m, 3);
    increases += _mm256_extract_epi16(m, 4);
    increases += _mm256_extract_epi16(m, 5);
    increases += _mm256_extract_epi16(m, 6);
    increases += _mm256_extract_epi16(m, 7);
    increases += _mm256_extract_epi16(m, 8);
    increases += _mm256_extract_epi16(m, 9);
    increases += _mm256_extract_epi16(m, 10);
    increases += _mm256_extract_epi16(m, 11);
    increases += _mm256_extract_epi16(m, 12);
    increases += _mm256_extract_epi16(m, 13);
    increases += _mm256_extract_epi16(m, 14);
    increases += _mm256_extract_epi16(m, 15);
    return increases;
}

static std::string deconstruct(const __m256i& m) {
    std::vector<int> v;
    v.push_back(_mm256_extract_epi16(m, 0));
    v.push_back(_mm256_extract_epi16(m, 1));
    v.push_back(_mm256_extract_epi16(m, 2));
    v.push_back(_mm256_extract_epi16(m, 3));
    v.push_back(_mm256_extract_epi16(m, 4));
    v.push_back(_mm256_extract_epi16(m, 5));
    v.push_back(_mm256_extract_epi16(m, 6));
    v.push_back(_mm256_extract_epi16(m, 7));
    v.push_back(_mm256_extract_epi16(m, 8));
    v.push_back(_mm256_extract_epi16(m, 9));
    v.push_back(_mm256_extract_epi16(m, 10));
    v.push_back(_mm256_extract_epi16(m, 11));
    v.push_back(_mm256_extract_epi16(m, 12));
    v.push_back(_mm256_extract_epi16(m, 13));
    v.push_back(_mm256_extract_epi16(m, 14));
    v.push_back(_mm256_extract_epi16(m, 15));

    std::stringstream s;
    for (int i : v) {
        s << std::setfill(' ') << std::setw(2) << i << " ";
    }
    return s.str();
}
static int avx(const unsigned short* depths, int n) {
    __m256i mask = _mm256_set_epi16(
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1
    );

    int i = 1;
    int aligned = n - (n % 16);

    __m256i incremental = _mm256_set1_epi16(0);
    for (; i < aligned; i += 16) {
        __m256i previous = _mm256_loadu_si256((__m256i*) &depths[i - 1]);
        __m256i current = _mm256_loadu_si256((__m256i*) &depths[i]);
        __m256i comparisons = _mm256_cmpgt_epi16(current, previous);
        __m256i masked = _mm256_and_si256(comparisons, mask);
        incremental = _mm256_add_epi32(incremental, masked);
        //std::cout << deconstruct(previous) << std::endl;
        //std::cout << deconstruct(current) << std::endl;
        //std::cout << deconstruct(comparisons) << std::endl;
        //std::cout << deconstruct(masked) << std::endl;
        //std::cout << deconstruct(incremental) << std::endl;
        //std::cout << sum_to_int(incremental) << std::endl;
    }

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
    //std::cout << "compiler: " << result << std::endl;
}

static void BM_avx(benchmark::State& state) {
    const auto depths = load_input("day1.input");

    int increases;
    for (auto _ : state) {
        increases = avx(depths.data(), depths.size());
        escape(&increases);
    }
    //std::cout << "avx: " << increases << std::endl;
}

BENCHMARK(BM_avx);
BENCHMARK(BM_manual_stride);
BENCHMARK(BM_compiler);

BENCHMARK_MAIN();
