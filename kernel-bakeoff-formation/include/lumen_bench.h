// lumen_bench.h: a small benchmark harness. Header-only, no dependencies.
//
//   #include "lumen_bench.h"
//   int main(int argc, char** argv) {
//       lumen::bench::Runner r(argc, argv);
//       r.run("example.n1000", /*items per op*/ 1000 * 60, /*seed*/ 42, [] { return my_op(1000, 60, 42); });
//       return r.result();
//   }
//
// An op must be pure: the same inputs every call, returning a checksum of its output. The harness
// checks that every call returns the same checksum (a determinism check) and prints it, so a later run can be compared
// against it. It warms up, sizes a sample to at least --min-sample-ms, takes --samples samples, and prints one JSON line
// per benchmark with the median and the MAD (median absolute deviation) of the per-op time.
//
// Further probes (not used in this folder, which calls only run(); a separate tool reads their rows):
// - Complexity slope: r.scaling(name, "n", {1000, 4000, 16000}, items_per_n, seed, [](uint64_t n) {...}) runs the op at
//   each size (rows "<name>.n<size>"); a reader of the rows can fit the log-log slope of time against n over the sizes, after dividing out
//   the declared complexity ("1", "n", "nlogn", "n2"), and flag it when it is steeper than declared.
// - Allocations: this header replaces global operator new and delete with counting versions, so every row carries
//   allocs_per_op, and r.steady(name, steps, setup, step) fails when a steady-state step allocates after its first call.
//   Include this header from exactly one .cpp per executable (the bench's), or define LUMEN_BENCH_NO_ALLOC_COUNT.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <new>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>  // __rdtsc for g++ and clang (untested)
#endif

namespace lumen::bench {

inline std::atomic<uint64_t> g_allocs{0};  // global operator new calls since start
inline uint64_t allocs() { return g_allocs.load(std::memory_order_relaxed); }

}  // namespace lumen::bench

// AddressSanitizer owns operator new/delete, so a replacement here fails to link (sanitizer builds). Under ASan the
// counters stay at zero: allocation assertions in sanitized test builds read 0, and benches never run sanitized.
#if defined(__SANITIZE_ADDRESS__) && !defined(LUMEN_BENCH_NO_ALLOC_COUNT)
#define LUMEN_BENCH_NO_ALLOC_COUNT
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) && !defined(LUMEN_BENCH_NO_ALLOC_COUNT)
#define LUMEN_BENCH_NO_ALLOC_COUNT
#endif
#endif

#ifndef LUMEN_BENCH_NO_ALLOC_COUNT
// Replacement allocation functions ([replacement.functions]): counted, then forwarded to the CRT. The nothrow forms
// default to calling these, so they are counted too.
void* operator new(std::size_t n) {
    lumen::bench::g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#ifdef _WIN32
void* operator new(std::size_t n, std::align_val_t a) {
    lumen::bench::g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = _aligned_malloc(n ? n : 1, std::size_t(a))) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void* p, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { _aligned_free(p); }
#endif
#endif

namespace lumen::bench {

inline double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

inline double mad(const std::vector<double>& v, double med) {
    std::vector<double> d;
    d.reserve(v.size());
    for (double x : v) d.push_back(x < med ? med - x : x - med);
    return median(d);
}

// Pin this thread to one performance core and raise its priority, to cut run-to-run noise. On a hybrid CPU (the
// i5-13600K has 6 P-cores and 8 E-cores) the scheduler may otherwise move the thread onto an E-core mid-run. Picks the
// last core of the highest efficiency class (core 0 takes most interrupts). Returns the logical processor, or -1.
inline int pin_to_fast_core() {
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    auto* base = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (!len || !GetLogicalProcessorInformationEx(RelationProcessorCore, base, &len)) return -1;
    int best = -1, bestClass = -1;
    for (DWORD off = 0; off < len;) {
        auto* p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        KAFFINITY m = p->Processor.GroupMask[0].Mask;
        if (p->Processor.GroupMask[0].Group == 0 && m && p->Processor.EfficiencyClass >= bestClass) {
            bestClass = p->Processor.EfficiencyClass;
            for (int i = 0; i < 64; ++i)
                if (m >> i & 1) { best = i; break; }
        }
        off += p->Size;
    }
    if (best < 0 || !SetThreadAffinityMask(GetCurrentThread(), KAFFINITY(1) << best)) return -1;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    return best;
#else
    return -1;
#endif
}

// One logical processor per physical core, split by efficiency class: P-cores (the highest class) and the rest (E-cores).
// SMT siblings are left out, so a worker count of k means k whole cores. {0, 0} where the OS cannot say.
struct CoreMasks { uint64_t p = 0, e = 0; };
inline CoreMasks core_masks() {
    CoreMasks out;
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    if (!len || !GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()), &len)) return out;
    int top = 0;
    for (DWORD off = 0; off < len;) {
        auto* p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        top = std::max<int>(top, p->Processor.EfficiencyClass); off += p->Size;
    }
    for (DWORD off = 0; off < len;) {
        auto* p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        uint64_t m = p->Processor.GroupMask[0].Group == 0 ? uint64_t(p->Processor.GroupMask[0].Mask) : 0;
        if (m) (p->Processor.EfficiencyClass == top ? out.p : out.e) |= m & (~m + 1);  // the core's lowest logical processor
        off += p->Size;
    }
#endif
    return out;
}

inline int popcount64(uint64_t m) { int n = 0; for (; m; m &= m - 1) ++n; return n; }

inline uint64_t tsc() {  // the invariant time-stamp counter: a fixed rate on current x86 CPUs, not the boosting core clock
#if defined(_MSC_VER) || defined(__x86_64__)
    return __rdtsc();
#else
    return 0;
#endif
}

class Runner {
public:
    int samples = 21;          // samples per benchmark; the median and MAD are over these
    double min_sample_ms = 2;  // each sample runs enough ops to last at least this long
    double warmup_ms = 50;     // untimed ops first: caches, branch predictors, clock ramp-up
    std::string filter;        // run only benchmarks whose name contains this
    int pinned = -1;

    Runner(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            auto arg = [&](const char* name) { return !std::strcmp(argv[i], name) && i + 1 < argc; };
            if (arg("--samples")) samples = std::max(1, std::atoi(argv[++i]));
            else if (arg("--min-sample-ms")) min_sample_ms = std::atof(argv[++i]);
            else if (arg("--warmup-ms")) warmup_ms = std::atof(argv[++i]);
            else if (arg("--filter")) filter = argv[++i];
        }
        pinned = pin_to_fast_core();
        counting_ = counting();
    }

    template <class Op>
    void run(const char* name, double items_per_op, uint64_t seed, Op op) { run_impl(name, items_per_op, seed, op, ""); }

    // The op at each size; complexity is "1", "n", "nlogn" or "n2", for fitting the slope over the sizes.
    template <class Op>
    void scaling(const char* name, const char* complexity, std::initializer_list<uint64_t> sizes, double items_per_n,
                 uint64_t seed, Op op) {
        for (uint64_t n : sizes) {
            char full[256], extra[256];
            std::snprintf(full, sizeof full, "%s.n%llu", name, (unsigned long long)n);
            std::snprintf(extra, sizeof extra, "\"scaling\":\"%s\",\"complexity\":\"%s\",\"size\":%llu,", name, complexity,
                          (unsigned long long)n);
            const double saved = working_set_bytes;
            if (working_set_per_n > 0) working_set_bytes = working_set_per_n * double(n);
            run_impl(full, items_per_n * double(n), seed, [&op, n] { return op(n); }, extra);
            working_set_bytes = saved;
        }
    }

    // A steady-state step must not allocate. setup() builds the state (untimed, may allocate); step(state) runs
    // `steps` times; any allocation after the first call is a finding (the first may grow buffers once).
    // The count is process-wide calls to global operator new over the probe: a background thread's allocations count, and a
    // step that allocates through malloc or its own pool is not seen.
    template <class Setup, class Step>
    void steady(const char* name, int steps, Setup setup, Step step) {
        if (!filter.empty() && std::string(name).find(filter) == std::string::npos) return;
        if (steps < 2) { std::fprintf(stderr, "%s: a steady-state probe needs 2 steps or more\n", name); failed_ = true; return; }
        if (!counting()) {  // never report a clean probe that could not have seen an allocation
            std::fprintf(stderr, "%s: allocation counting is off in this build (LUMEN_BENCH_NO_ALLOC_COUNT?), so the allocation probe cannot pass\n", name);
            failed_ = true; return;
        }
        auto state = setup();
        uint64_t a0 = allocs();
        step(state);
        uint64_t a1 = allocs();
        for (int i = 1; i < steps; ++i) step(state);
        uint64_t later = allocs() - a1;
        std::printf("{\"steady\":\"%s\",\"steps\":%d,\"first_step_allocs\":%llu,\"steady_allocs\":%llu,\"ok\":%s}\n", name, steps,
                    (unsigned long long)(a1 - a0), (unsigned long long)later, later ? "false" : "true");
        std::fflush(stdout);
        if (later) { std::fprintf(stderr, "%s: %llu allocations in %d steady-state steps\n", name, (unsigned long long)later, steps - 1); failed_ = true; }
    }

    // Bytes the next runs touch per item (reads plus writes), declared by the bench, for setting against the
    // machine's measured bandwidth (a stream benchmark) to class each run as memory- or compute-bound. 0 means undeclared.
    double bytes_per_item = 0;
    // The bytes the op keeps live (its arrays), for picking the roof of the cache level that holds them (P-core
    // L1, L2, L3 or DRAM). For r.scaling(), working_set_per_n * n is used at each size when it is set. 0 means undeclared.
    double working_set_bytes = 0;
    double working_set_per_n = 0;

    // op(workers) at each worker count with the process on one logical processor per P-core (SMT siblings left out),
    // then all P- and E-cores with one worker per core. Rows "<name>.w<k>" carry workers and cores ("P" or "P+E"), for the parallel
    // efficiency against one worker; it fails when the checksum depends on the worker count.
    template <class Op>
    void threads(const char* name, std::initializer_list<int> workers, double items_per_op, uint64_t seed, Op op) {
        const CoreMasks cm = core_masks();
#ifdef _WIN32
        DWORD_PTR proc = 0, sys = 0;
        GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);
        auto set = [](uint64_t m) { if (m) SetProcessAffinityMask(GetCurrentProcess(), DWORD_PTR(m)); };
#else
        auto set = [](uint64_t) {};
#endif
        auto one = [&](int k, const char* cores, const char* tag) {
            char full[256], extra[256];
            std::snprintf(full, sizeof full, "%s.%s%d", name, tag, k);  // .w<k> on P-cores, .pe<k> on P+E: never the same row
            std::snprintf(extra, sizeof extra, "\"threads\":\"%s\",\"workers\":%d,\"cores\":\"%s\",", name, k, cores);
            run_impl(full, items_per_op, seed, [&op, k] { return op(k); }, extra);
        };
        set(cm.p);
        for (int k : workers) one(k, "P", "w");
        if (cm.e) { set(cm.p | cm.e); one(popcount64(cm.p | cm.e), "P+E", "pe"); }
#ifdef _WIN32
        SetProcessAffinityMask(GetCurrentProcess(), proc);
#endif
        pinned = pin_to_fast_core();  // a process mask change resets this thread's affinity: pin it again for later runs
    }

    int result() const { return failed_ ? 1 : 0; }

    // Whether global operator new reaches the counting replacement. A direct call to ::operator new is a function call,
    // which the compiler may not elide the way it may elide a new-expression.
    static bool counting() {
        uint64_t c0 = allocs();
        void* p = ::operator new(1);
        ::operator delete(p);
        return allocs() != c0;
    }

private:
    template <class Op>
    void run_impl(const char* name, double items_per_op, uint64_t seed, Op op, const char* extra) {
        if (!filter.empty() && std::string(name).find(filter) == std::string::npos) return;
        using clock = std::chrono::steady_clock;
        auto ms_since = [](clock::time_point t) { return std::chrono::duration<double, std::milli>(clock::now() - t).count(); };
        const uint64_t checksum = op();
        bool stable = true;
        auto call = [&] { uint64_t c = op(); sink_ ^= c; stable &= c == checksum; };

        uint64_t iters = 1;  // warm up while sizing a sample: double the batch until it lasts min_sample_ms
        for (auto start = clock::now();;) {
            auto t = clock::now();
            for (uint64_t i = 0; i < iters; ++i) call();
            double ms = ms_since(t);
            if (ms >= min_sample_ms && ms_since(start) >= warmup_ms) break;
            if (ms < min_sample_ms) iters *= 2;
        }
        std::vector<double> ns;
        ns.reserve(samples);
        const uint64_t a0 = allocs();
        const uint64_t tsc0 = tsc();
        for (int s = 0; s < samples; ++s) {
            auto t = clock::now();
            for (uint64_t i = 0; i < iters; ++i) call();
            ns.push_back(ms_since(t) * 1e6 / double(iters));
        }
        const double ops = double(iters) * samples;
        const double tsc_per_item = items_per_op > 0 ? double(tsc() - tsc0) / (ops * items_per_op) : 0;  // TSC ticks, not core cycles
        const double allocs_per_op = double(allocs() - a0) / ops;  // the list below allocates after this
        double med = median(ns), dev = mad(ns, med);
        auto [lo, hi] = std::minmax_element(ns.begin(), ns.end());
        std::string list;
        for (double x : ns) { char b[32]; std::snprintf(b, sizeof b, "%s%.1f", list.empty() ? "" : ",", x); list += b; }
        std::printf("{\"bench\":\"%s\",%s\"n\":%d,\"iters\":%llu,\"median_ns\":%.1f,\"mad_ns\":%.1f,\"min_ns\":%.1f,\"max_ns\":%.1f,"
                    "\"items\":%.0f,\"items_per_s\":%.0f,\"seed\":%llu,\"checksum\":\"%016llx\",\"stable\":%s,\"pinned\":%d,"
                    "\"allocs_per_op\":%.3f,\"alloc_counting\":%s,\"tsc_per_item\":%.3f,\"bytes_per_item\":%.1f,"
                    "\"working_set_bytes\":%.0f,\"samples_ns\":[%s]}\n",
                    name, extra, samples, (unsigned long long)iters, med, dev, *lo, *hi, items_per_op,
                    med > 0 ? items_per_op * 1e9 / med : 0, (unsigned long long)seed, (unsigned long long)checksum,
                    stable ? "true" : "false", pinned, allocs_per_op, counting_ ? "true" : "false", tsc_per_item, bytes_per_item,
                    working_set_bytes, list.c_str());
        std::fflush(stdout);
        if (!stable) { std::fprintf(stderr, "%s: the op returned different checksums across calls (not deterministic)\n", name); failed_ = true; }
    }

    bool counting_ = false;  // rows say whether allocs_per_op was measured, and so whether the atomic count was in the timing
    volatile uint64_t sink_ = 0;  // every checksum is written here: a volatile write keeps the ops from being optimised away
    bool failed_ = false;
};

}  // namespace lumen::bench
