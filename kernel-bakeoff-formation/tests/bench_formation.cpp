// lumen::formation benchmarks: the one-off cost of a formation command at 100,000 agents (shape slots, then the rank-split
// assignment). Agents: 100,000 uniform in a 560 m square.
//   py -3 kbuild.py original --bench
// Declared complexity: slots O(grid cells), assign O(n log n).
// The original file also ran two swarm-step benchmarks after these two; they need the swarm core and are left out here.
#include "formation.h"
#include "lumen_bench.h"

#include <random>
#include <vector>

using namespace lumen::formation;

int main(int argc, char** argv) {
    lumen::bench::Runner r(argc, argv);
    const std::size_t n = 100000;
    std::mt19937 g(5);
    std::vector<float> ax(n), ay(n), sx, sy;
    for (std::size_t i = 0; i < n; ++i) { ax[i] = float(g() % 56000); ay[i] = float(g() % 56000); }
    r.run("formation.slots.dragon.n100000", double(n), 5, [&] {
        shape_slots("dragon", n, 70.f, 28000.f, 28000.f, 0.f, 1.f, sx, sy);
        return std::uint64_t(sx.size());
    });
    std::vector<std::uint32_t> slot;
    r.run("formation.assign.n100000", double(n), 5, [&] {
        assign(ax, ay, sx, sy, slot);
        return std::uint64_t(slot[0]) * 131 + slot[n - 1];
    });
    return r.result();
}
