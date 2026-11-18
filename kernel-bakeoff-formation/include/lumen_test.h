// lumen_test.h: the smallest test runner for engine-free cores. Header-only.
//
//   #include "lumen_test.h"
//   TEST(replay_is_identical) { CHECK_EQ(run(64, 600, 1), run(64, 600, 1)); }
//   int main() { return lumen::test::run_all(); }
#pragma once
#include <cstdio>
#include <functional>
#include <utility>
#include <vector>

namespace lumen::test {
struct Case { const char* name; void (*fn)(); };
inline std::vector<Case>& cases() { static std::vector<Case> c; return c; }
inline int& failures() { static int f = 0; return f; }
struct Register { Register(const char* n, void (*f)()) { cases().push_back({n, f}); } };

inline int run_all() {
    int failed = 0;
    for (auto& c : cases()) {
        int before = failures();
        c.fn();
        bool ok = failures() == before;
        failed += !ok;
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", c.name);
    }
    std::printf("%zu tests, %d failed\n", cases().size(), failed);
    return failed || cases().empty() ? 1 : 0;  // a test file with no tests is a mistake, not a pass
}
}  // namespace lumen::test

#define TEST(name)                                                       \
    static void name();                                                  \
    static lumen::test::Register name##_reg(#name, name);                \
    static void name()
#define CHECK(cond)                                                                                        \
    do { if (!(cond)) { ++lumen::test::failures(); std::printf("  %s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); } } while (0)
// A replay hash pinned per compiler. Float results can differ between compiler versions (the CRT's sqrt, sin and the like
// changed between VS 2019 and 2022; spatial's hash differs, hello's doesn't), so a pin names the compiler it was recorded
// with: CHECK_PINNED(run(...), {{1944, 0x2cec783cc308afd7ull}, {1929, 0x...ull}}). The key is _MSC_VER plus the
// toolset's minor (19.44 -> 1944, 19.29 -> 1929); __clang__ builds use 0. With no pin for this compiler the check prints
// SKIP and passes: record one by running the test there.
namespace lumen::test {
inline int compiler_key() {
#if defined(__clang__)
    return 0;
#elif defined(_MSC_VER)
    return _MSC_VER;  // 1944 for 19.44, 1929 for 19.29
#else
    return -1;
#endif
}
}  // namespace lumen::test
#define CHECK_PINNED(value, ...)                                                                                                 \
    do {                                                                                                                         \
        const std::pair<int, unsigned long long> pins_[] = __VA_ARGS__;                                                          \
        unsigned long long v_ = (value); bool found_ = false;                                                                    \
        for (const auto& p_ : pins_) if (p_.first == lumen::test::compiler_key()) {                                               \
            found_ = true;                                                                                                       \
            if (v_ != p_.second) { ++lumen::test::failures();                                                                    \
                std::printf("  %s:%d: CHECK_PINNED(%s): %016llx, pinned %016llx for compiler %d\n", __FILE__, __LINE__, #value, v_, \
                            p_.second, p_.first); } }                                                                            \
        if (!found_) std::printf("  SKIP %s:%d: no pin for compiler %d (value %016llx)\n", __FILE__, __LINE__,                    \
                                 lumen::test::compiler_key(), v_);                                                               \
    } while (0)
#define CHECK_EQ(a, b)                                                                                              \
    do { auto va_ = (a); auto vb_ = (b); if (!(va_ == vb_)) { ++lumen::test::failures();                               \
        std::printf("  %s:%d: CHECK_EQ(%s, %s): %llu != %llu\n", __FILE__, __LINE__, #a, #b, (unsigned long long)va_, \
                    (unsigned long long)vb_); } } while (0)
