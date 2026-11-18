// lumen::formation tests: the slots fill a shape with exactly n points at the asked spacing, and the rank-split assignment
// is a bijection that beats a shuffled one by far and is deterministic.
//   py -3 kbuild.py original
// The original file also had a third test (a swarm pours into the dragon); it needs the swarm core and is left out here.
#include "formation.h"
#include "lumen_test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>

using namespace lumen::formation;

namespace {
double cost(std::span<const float> ax, std::span<const float> ay, std::span<const float> sx, std::span<const float> sy, const std::vector<std::uint32_t>& slot) {
    double c = 0;
    for (std::size_t i = 0; i < ax.size(); ++i) c += std::hypot(double(ax[i]) - sx[slot[i]], double(ay[i]) - sy[slot[i]]);
    return c / double(ax.size());
}
}  // namespace

TEST(slots_fill_a_shape_with_exactly_n) {
    std::vector<float> sx, sy;
    for (const char* name : {"dragon", "sword", "hand", "disk"}) {
        CHECK(shape_slots(name, 20000, 80.f, 1000.f, 2000.f, 0.f, 1.f, sx, sy));
        CHECK_EQ(sx.size(), std::size_t(20000));
        // nearest-neighbour spacing on the grid is the asked 80 cm (spot check: slot 0 to its nearest)
        float best = 1e30f;
        for (std::size_t j = 1; j < sx.size(); ++j) best = std::min(best, std::hypot(sx[j] - sx[0], sy[j] - sy[0]));
        CHECK(best > 79.f && best < 81.f);
    }
    CHECK(!shape_slots("teapot", 100, 80.f, 0.f, 0.f, 0.f, 1.f, sx, sy));
}

TEST(rank_split_is_a_bijection_and_far_better_than_shuffled) {
    const std::size_t n = 20000;
    std::mt19937 g(3);
    std::vector<float> ax(n), ay(n), sx, sy;
    for (std::size_t i = 0; i < n; ++i) { ax[i] = float(g() % 20000); ay[i] = float(g() % 20000); }
    CHECK(shape_slots("dragon", n, 80.f, 10000.f, 10000.f, 0.f, 1.f, sx, sy));
    std::vector<std::uint32_t> slot;
    assign(ax, ay, sx, sy, slot);
    std::vector<std::uint32_t> sorted = slot;
    std::sort(sorted.begin(), sorted.end());
    bool perm = true;
    for (std::size_t i = 0; i < n; ++i) perm &= sorted[i] == i;
    CHECK(perm);
    std::vector<std::uint32_t> shuffled(n);
    std::iota(shuffled.begin(), shuffled.end(), 0u);
    std::shuffle(shuffled.begin(), shuffled.end(), g);
    const double c = cost(ax, ay, sx, sy, slot), r = cost(ax, ay, sx, sy, shuffled);
    std::printf("  mean walk: rank split %.0f cm, shuffled %.0f cm\n", c, r);
    CHECK(c < 0.5 * r);
    std::vector<std::uint32_t> again;
    assign(ax, ay, sx, sy, again);
    CHECK(again == slot);  // deterministic
}

int main() { return lumen::test::run_all(); }
