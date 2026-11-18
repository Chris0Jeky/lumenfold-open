// Exactness check: lumen::formation (the build under test) against ref_formation (the original, renamed), bit for
// bit, over a fixed input set. shape_slots: the return value and every sx, sy float's bits. assign: every slot_of entry, and
// that a length mismatch throws std::invalid_argument. Exit 0 only when all cases match.
#include "formation.h"
#include "ref_formation.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static int cases = 0, bad = 0;
static void fail(const char* what, std::size_t n, int variant) { ++bad; if (bad <= 20) std::printf("MISMATCH %s n=%zu variant=%d\n", what, n, variant); }
static bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

int main() {
    const char* shapes[] = {"dragon", "sword", "hand", "disk", "teapot", ""};
    const std::size_t ns[] = {0, 1, 2, 3, 7, 64, 999, 5000, 20000, 100000};
    const float sp[][5] = {{70.f, 28000.f, 28000.f, 0.f, 1.f}, {80.f, 1000.f, 2000.f, 0.f, 1.f}, {33.3f, -500.f, 12.5f, 0.6f, 0.8f},
                           {125.f, 0.f, 0.f, -0.70710677f, 0.70710677f}, {0.f, 0.f, 0.f, 0.f, 1.f}, {-5.f, 0.f, 0.f, 0.f, 1.f}};
    for (const char* s : shapes)
        for (std::size_t n : ns)
            for (int v = 0; v < 6; ++v) {
                std::vector<float> ax_, ay_, bx_, by_; ax_.push_back(9.f); bx_.push_back(7.f);
                const bool ra = lumen::formation::shape_slots(s, n, sp[v][0], sp[v][1], sp[v][2], sp[v][3], sp[v][4], ax_, ay_);
                const bool rb = ref_formation::shape_slots(s, n, sp[v][0], sp[v][1], sp[v][2], sp[v][3], sp[v][4], bx_, by_);
                ++cases;
                if (ra != rb || !same(ax_, bx_) || !same(ay_, by_)) fail(s, n, v);
            }
    // assign: uniform, clustered, integer grid with ties, all one point, huge and negative, and shape slots as targets
    const std::size_t an[] = {0, 1, 2, 3, 5, 17, 256, 1000, 4097, 30000, 100000};
    for (std::size_t n : an)
        for (int v = 0; v < 7; ++v) {
            std::mt19937 g(unsigned(n * 7 + v));
            std::vector<float> ax(n), ay(n), sx(n), sy(n);
            for (std::size_t i = 0; i < n; ++i) {
                switch (v) {
                    case 0: ax[i] = float(g() % 56000); ay[i] = float(g() % 56000); sx[i] = float(g() % 56000); sy[i] = float(g() % 56000); break;
                    case 1: ax[i] = float(g() % 16); ay[i] = float(g() % 16); sx[i] = float(g() % 16); sy[i] = float(g() % 16); break;
                    case 2: ax[i] = ay[i] = sx[i] = sy[i] = 5.f; break;
                    case 3: { std::normal_distribution<float> d(0.f, 1e6f); ax[i] = d(g); ay[i] = d(g) * 1e-3f; sx[i] = -d(g); sy[i] = d(g); } break;
                    case 4: ax[i] = float(i % 2) * 1e-30f; ay[i] = -0.f; sx[i] = 0.f; sy[i] = float(i % 3); break;
                    case 5: { std::uniform_real_distribution<float> d(-1.f, 1.f); ax[i] = d(g); ay[i] = d(g) * 4.f; sx[i] = d(g) * 4.f; sy[i] = d(g); } break;
                    case 6: ax[i] = float(g() % 56000); ay[i] = float(g() % 56000); break;
                }
            }
            if (v == 6 && n) { std::vector<float> tx, ty;
                if (ref_formation::shape_slots("dragon", n, 70.f, 28000.f, 28000.f, 0.f, 1.f, tx, ty)) { sx = tx; sy = ty; } }
            std::vector<std::uint32_t> a{1, 2}, b{3};
            lumen::formation::assign(ax, ay, sx, sy, a);
            ref_formation::assign(ax, ay, sx, sy, b);
            ++cases;
            if (a != b) fail("assign", n, v);
        }
    {   // a length mismatch throws in both
        std::vector<float> x(4), y(3); std::vector<std::uint32_t> o;
        bool threw = false;
        try { lumen::formation::assign(x, x, x, y, o); } catch (const std::invalid_argument&) { threw = true; }
        ++cases; if (!threw) fail("assign-throws", 4, 0);
    }
    std::printf("exact_formation: %d cases, %d mismatches\n", cases, bad);
    return bad ? 1 : 0;
}
