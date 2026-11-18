// Wider exactness check (not shown to the models during the task): random shapes, counts, spacings, centres and
// rotations for shape_slots, and random point sets of many kinds for assign (ties, duplicates, signed zeros, subnormals,
// infinities, huge and tiny values, aliasing of sx/sy into both calls), against the renamed original. NaN is left out:
// the original's comparator is not a strict weak order with NaN, so its result is not defined. Exit 0 only on no mismatch.
#include "formation.h"
#include "ref_formation.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static int cases = 0, bad = 0;
static bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

int main() {
    std::mt19937_64 g(0xC0B2u);
    const char* shapes[] = {"dragon", "sword", "hand", "disk"};
    auto U = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(g); };
    for (int t = 0; t < 600; ++t) {
        const char* s = shapes[g() % 4];
        std::size_t n = (t % 5 == 0) ? std::size_t(g() % 200000 + 1) : std::size_t(g() % 5000 + 1);
        float sp = float(U(0.5, 400.0)), cx = float(U(-1e6, 1e6)), cy = float(U(-1e6, 1e6));
        double yaw = U(-3.141592653589793, 3.141592653589793);
        float c = float(std::cos(yaw)), sn = float(std::sin(yaw));
        if (t % 7 == 0) { c = float(U(-3, 3)); sn = float(U(-3, 3)); }  // not a unit vector: the contract still says bit-identical
        std::vector<float> ax_, ay_, bx_, by_;
        const bool ra = lumen::formation::shape_slots(s, n, sp, cx, cy, c, sn, ax_, ay_);
        const bool rb = ref_formation::shape_slots(s, n, sp, cx, cy, c, sn, bx_, by_);
        ++cases;
        if (ra != rb || !same(ax_, bx_) || !same(ay_, by_)) { if (++bad <= 20) std::printf("MISMATCH slots %s n=%zu t=%d\n", s, n, t); }
    }
    const float inf = std::numeric_limits<float>::infinity(), den = std::numeric_limits<float>::denorm_min();
    for (int t = 0; t < 900; ++t) {
        std::size_t n = (t % 9 == 0) ? std::size_t(g() % 150000) : std::size_t(g() % 3000);
        std::vector<float> ax(n), ay(n), sx(n), sy(n);
        const int kind = t % 9;
        for (std::size_t i = 0; i < n; ++i) {
            float* v[4] = {&ax[i], &ay[i], &sx[i], &sy[i]};
            for (float* p : v) {
                switch (kind) {
                    case 0: *p = float(U(-1e4, 1e4)); break;
                    case 1: *p = float(int(g() % 7) - 3); break;                                  // heavy ties
                    case 2: *p = (g() % 2 ? -0.f : 0.f) + float(int(g() % 3)) * den; break;        // signed zeros, subnormals
                    case 3: *p = (g() % 50 == 0) ? (g() % 2 ? inf : -inf) : float(U(-1, 1)); break; // infinities
                    case 4: *p = float(U(-1, 1) * std::pow(10.0, U(-30, 30))); break;              // wide exponents
                    case 5: *p = float(g() % 56000); break;                                         // the bench's kind
                    case 6: *p = float(U(0, 1e-3)) + 1e6f; break;                                    // big, nearly equal
                    case 7: *p = i % 2 ? float(i) : -float(i); break;
                    default: *p = float(U(-560, 560)); break;
                }
            }
        }
        std::vector<std::uint32_t> a, b;
        lumen::formation::assign(ax, ay, sx, sy, a); ref_formation::assign(ax, ay, sx, sy, b);
        ++cases;
        if (a != b) { if (++bad <= 40) std::printf("MISMATCH assign kind=%d n=%zu t=%d\n", kind, n, t); }
        if (t % 50 == 0) {  // the same spans passed twice (agents are slots)
            lumen::formation::assign(ax, ay, ax, ay, a); ref_formation::assign(ax, ay, ax, ay, b);
            ++cases; if (a != b) { if (++bad <= 40) std::printf("MISMATCH assign-self n=%zu t=%d\n", n, t); }
        }
    }
    std::printf("exact_wide: %d cases, %d mismatches\n", cases, bad);
    return bad ? 1 : 0;
}
