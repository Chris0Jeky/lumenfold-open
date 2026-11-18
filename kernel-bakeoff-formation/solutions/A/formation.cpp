// lumen::formation (see formation.h).
#include "formation.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lumen::formation {

namespace {

// The silhouettes (original outlines), in [-960, 960] units.
int shape_id(std::string_view s) { return s == "disk" ? 0 : s == "sword" ? 1 : s == "hand" ? 2 : s == "dragon" ? 3 : -1; }
struct RowSpan { int j, lo, hi; };  // half-open grid-index interval
struct Interval { int lo, hi; };

template<class Pred>
int first_true_near(int lo, int hi, double estimate, Pred pred) {
    // An analytic intersection is only a starting guess. Correct it with the original predicate,
    // so even a rounded intersection on the wrong side of a grid point cannot alter membership.
    int p = int(std::clamp(estimate, double(lo), double(hi)));
    while (p > lo && pred(p - 1)) --p;
    while (p < hi && !pred(p)) ++p;
    return p;
}

std::size_t count(int id, double u, std::vector<RowSpan>* rows = nullptr) {
    if (rows) rows->clear();
    std::size_t count = 0;
    const int k = int(std::floor(960.0 / u));
    for (int j = -k; j <= k; ++j) {
        const double y = j * u;
        std::array<Interval, 11> spans;  // at most eleven silhouette primitives
        int used = 0;
        auto add = [&](int lo, int hi) { if (lo < hi) spans[used++] = {lo, hi}; };
        auto rect = [&](double x0, double y0, double x1, double y1) {
            if (y < y0 || y > y1) return;
            const int lo = first_true_near(-k, k + 1, x0 / u, [&](int i) { return i * u >= x0; });
            const int hi = first_true_near(lo, k + 1, x1 / u, [&](int i) { return i * u > x1; });
            add(lo, hi);
        };
        auto oval = [&](double cx, double cy, double rx, double ry) {
            const double dy = (y - cy) / ry, dy2 = dy * dy;
            if (dy2 > 1.0) return;
            auto in = [&](int i) {
                const double dx = (i * u - cx) / rx;
                return dx * dx + dy2 <= 1.0;
            };
            // One of the two grid points bracketing the centre must be inside if this row is nonempty.
            int p = first_true_near(-k, k + 1, cx / u, [&](int i) { return i * u >= cx; });
            if (p > k || !in(p)) { if (p == -k || !in(p - 1)) return; --p; }
            const double radius = rx * std::sqrt(1.0 - dy2);
            const int lo = first_true_near(-k, p, (cx - radius) / u, in);
            const int hi = first_true_near(p + 1, k + 1, (cx + radius) / u, [&](int i) { return !in(i); });
            add(lo, hi);
        };
        auto tri = [&](double ax, double ay, double bx, double by, double cx, double cy) {
            // These fixed integer triangles have |twice-area| >= 10100; cross-product roundoff on this
            // bounded grid is < 1e-8. The opposite winding cannot satisfy all three original inequalities.
            const bool positive = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0;
            int lo = -k, hi = k + 1;
            auto edge = [&](double ux, double uy, double vx, double vy) {
                if (lo == hi) return;
                const double a = (vx - ux) * (y - uy), b = vy - uy;
                auto accepts = [&](int i) {
                    const double c = a - b * (i * u - ux);
                    return positive ? c >= 0 : c <= 0;
                };
                if (b == 0) { if (!accepts(lo)) hi = lo; }
                else if ((b < 0) == positive) lo = first_true_near(lo, hi, (ux + a / b) / u, accepts);
                else hi = first_true_near(lo, hi, (ux + a / b) / u, [&](int i) { return !accepts(i); });
            };
            edge(ax, ay, bx, by); edge(bx, by, cx, cy); edge(cx, cy, ax, ay);
            add(lo, hi);
        };
        switch (id) {
            case 0: oval(0, 0, 820, 820); break;
            case 1:
                rect(-100, -330, 100, 700); tri(-100, 700, 100, 700, 0, 960); rect(-440, -420, 440, -330);
                rect(-60, -800, 60, -420); oval(0, -835, 115, 85); break;
            case 2:
                rect(-360, -450, 330, 230); oval(0, -450, 360, 240); oval(-280, 380, 90, 380);
                oval(-85, 480, 90, 440); oval(110, 460, 90, 420); oval(300, 340, 85, 330);
                tri(-350, -230, -700, 170, -500, 320); oval(-570, 180, 125, 140); break;
            case 3:
                oval(100, -100, 280, 460); oval(300, 430, 180, 240); rect(340, 430, 660, 530);
                tri(270, 600, 340, 850, 420, 570); tri(-100, 100, -900, 740, -700, 70);
                tri(-100, 100, -700, 70, -590, -280); tri(120, 150, 850, 790, 730, 110);
                tri(120, 150, 730, 110, 650, -250); tri(50, -350, -820, -780, -230, -500);
                tri(-820, -780, -900, -510, -650, -660); tri(80, -380, -40, -700, 250, -620); break;
        }
        std::sort(spans.begin(), spans.begin() + used, [](Interval a, Interval b) { return a.lo < b.lo; });
        for (int a = 0; a < used;) {
            Interval merged = spans[a++];
            while (a < used && spans[a].lo <= merged.hi) merged.hi = std::max(merged.hi, spans[a++].hi);
            count += std::size_t(merged.hi - merged.lo);
            if (rows) rows->push_back({j, merged.lo, merged.hi});
        }
    }
    return count;
}

}  // namespace

bool shape_slots(std::string_view shape, std::size_t n, float spacing, float cx, float cy, float cos_yaw, float sin_yaw,
                 std::vector<float>& sx, std::vector<float>& sy) {
    sx.clear(); sy.clear();
    const int id = shape_id(shape);
    if (id < 0 || n == 0 || !(std::isfinite(spacing) && spacing > 0.f)) return false;
    // A grid step u (units) with at least n slots inside: estimate from the fill at a coarse step, then shrink 1% at a time.
    const double fill = double(count(id, 16.0)) * 256.0;  // area in square units
    double u = std::sqrt(fill / double(n)) * 1.02;
    std::size_t m = 0;
    std::vector<RowSpan> rows;
    for (int tries = 0; tries < 400 && u >= 0.25; ++tries, u *= 0.99)
        if ((m = count(id, u, &rows)) >= n) break;
    if (m < n) return false;
    // Keep n of the m inside points, spread evenly in row-major order, as world positions.
    const double s = double(spacing) / u;  // cm per unit
    std::uint64_t remainder = 0;
    sx.reserve(n); sy.reserve(n);
    for (const RowSpan row : rows)
        for (int i = row.lo; i < row.hi; ++i) {
            const double x = i * u, y = row.j * u;
            // Quotient differences by remainder accumulation. n <= m and k <= 3840, so this cannot overflow.
            remainder += n;
            if (remainder >= m) {
                remainder -= m;
                const double wx = s * x, wy = s * y;  // shape +y along (cos_yaw, sin_yaw), +x to its right
                sx.push_back(float(cx + wy * cos_yaw + wx * sin_yaw));
                sy.push_back(float(cy + wy * sin_yaw - wx * cos_yaw));
            }
        }
    return sx.size() == n;
}

namespace {

// Keep the original algorithm for small inputs and for coordinates without a total numeric order (NaNs).
void assign_original(std::span<const float> ax, std::span<const float> ay, std::span<const float> sx, std::span<const float> sy,
            std::vector<std::uint32_t>& slot_of) {
    const std::size_t n = ax.size();
    if (ay.size() != n || sx.size() != n || sy.size() != n) throw std::invalid_argument("lumen::formation::assign: one length for agents and slots");
    std::vector<std::uint32_t> A(n), S(n);
    for (std::size_t i = 0; i < n; ++i) { A[i] = std::uint32_t(i); S[i] = std::uint32_t(i); }
    slot_of.assign(n, 0);
    std::vector<std::pair<std::size_t, std::size_t>> stack;
    if (n) stack.push_back({0, n});
    while (!stack.empty()) {
        const auto [lo, hi] = stack.back(); stack.pop_back();
        if (hi - lo == 1) { slot_of[A[lo]] = S[lo]; continue; }
        float x0 = ax[A[lo]], x1 = x0, y0 = ay[A[lo]], y1 = y0;
        for (std::size_t k = lo; k < hi; ++k) {
            x0 = std::min({x0, ax[A[k]], sx[S[k]]}); x1 = std::max({x1, ax[A[k]], sx[S[k]]});
            y0 = std::min({y0, ay[A[k]], sy[S[k]]}); y1 = std::max({y1, ay[A[k]], sy[S[k]]});
        }
        const bool by_x = x1 - x0 >= y1 - y0;
        const std::size_t mid = lo + (hi - lo) / 2;
        auto split = [&](std::vector<std::uint32_t>& v, std::span<const float> px, std::span<const float> py) {
            std::nth_element(v.begin() + std::ptrdiff_t(lo), v.begin() + std::ptrdiff_t(mid), v.begin() + std::ptrdiff_t(hi),
                [&](std::uint32_t a, std::uint32_t b) {
                    const float ka = by_x ? px[a] : py[a], kb = by_x ? px[b] : py[b];
                    return ka != kb ? ka < kb : a < b;
                });
        };
        split(A, ax, ay); split(S, sx, sy);
        stack.push_back({mid, hi}); stack.push_back({lo, mid});
    }
}

std::uint32_t float_key(float v) {
    // IEEE finite numeric order, with both signed zeros tied exactly as in the original comparator.
    const std::uint32_t bits = v == 0.f ? 0u : std::bit_cast<std::uint32_t>(v);
    return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}

void sort_axis(std::span<const float> p, std::vector<std::uint32_t>& order, std::vector<std::uint32_t>& scratch,
               std::vector<std::uint32_t>& rank) {
    const std::size_t n = p.size();
    for (std::size_t i = 0; i < n; ++i) order[i] = std::uint32_t(i);
    // Random inputs usually fail at their first few elements; already-sorted axes need no radix passes.
    if (std::is_sorted(p.begin(), p.end())) {
        for (std::size_t i = 0; i < n; ++i) rank[i] = std::uint32_t(i);
        return;
    }
    // The caller limits n to UINT32_MAX, so cumulative counts fit in 32 bits as well.
    std::uint32_t hist[4][256]{};
    for (std::size_t i = 0; i < n; ++i) {
        const auto key = rank[i] = float_key(p[i]);
        for (int b = 0; b < 4; ++b) ++hist[b][(key >> (8 * b)) & 255];
    }
    for (int b = 0; b < 4; ++b) {
        // A uniform byte contributes no ordering information; skipping its stable pass is exact.
        if (hist[b][(rank[0] >> (8 * b)) & 255] == n) continue;
        std::uint32_t sum = 0;
        for (auto& h : hist[b]) { const auto old = h; h = sum; sum += old; }
        // Stable passes start in index order, so equal coordinates always break ties by index.
        for (auto index : order) scratch[hist[b][(rank[index] >> (8 * b)) & 255]++] = index;
        order.swap(scratch);
    }
    // A global rank compares identically within every recursive subset, including coordinate ties.
    for (std::size_t i = 0; i < n; ++i) rank[order[i]] = std::uint32_t(i);
}

}  // namespace

void assign(std::span<const float> ax, std::span<const float> ay, std::span<const float> sx, std::span<const float> sy,
            std::vector<std::uint32_t>& slot_of) {
    const std::size_t n = ax.size();
    if (ay.size() != n || sx.size() != n || sy.size() != n) throw std::invalid_argument("lumen::formation::assign: one length for agents and slots");
    if (n < 64 || n > std::numeric_limits<std::uint32_t>::max()) { assign_original(ax, ay, sx, sy, slot_of); return; }
    for (std::size_t i = 0; i < n; ++i)
        if (!(std::isfinite(ax[i]) && std::isfinite(ay[i]) && std::isfinite(sx[i]) && std::isfinite(sy[i]))) {
            assign_original(ax, ay, sx, sy, slot_of); return;
        }
    slot_of.assign(n, 0);
    std::vector<std::uint32_t> a_x(n), a_y(n), s_x(n), s_y(n), scratch(n);
    std::vector<std::uint32_t> ar_x(n), ar_y(n), sr_x(n), sr_y(n);
    sort_axis(ax, a_x, scratch, ar_x); sort_axis(ay, a_y, scratch, ar_y);
    sort_axis(sx, s_x, scratch, sr_x); sort_axis(sy, s_y, scratch, sr_y);
    auto visit = [&](auto&& self, std::size_t lo, std::size_t hi) -> void {
        if (hi - lo == 1) { slot_of[a_x[lo]] = s_x[lo]; return; }
        // Sorted endpoints give the same numeric extrema as a scan. Signed-zero choices cannot change >= of widths.
        const float x0 = std::min(ax[a_x[lo]], sx[s_x[lo]]), x1 = std::max(ax[a_x[hi - 1]], sx[s_x[hi - 1]]);
        const float y0 = std::min(ay[a_y[lo]], sy[s_y[lo]]), y1 = std::max(ay[a_y[hi - 1]], sy[s_y[hi - 1]]);
        const bool by_x = x1 - x0 >= y1 - y0;
        const std::size_t mid = lo + (hi - lo) / 2;
        if (hi - lo == 2) {
            const auto& a = by_x ? a_x : a_y;
            const auto& s = by_x ? s_x : s_y;
            slot_of[a[lo]] = s[lo]; slot_of[a[mid]] = s[mid];
            return;
        }
        auto split = [&](const std::vector<std::uint32_t>& primary, std::vector<std::uint32_t>& secondary,
                         const std::vector<std::uint32_t>& rank) {
            const auto key = rank[primary[mid]];
            std::size_t left = lo, right = mid;
            for (std::size_t i = lo; i < hi; ++i) {
                const auto index = secondary[i];
                const bool lower = rank[index] < key;
                const auto dest = lower ? left : right;
                scratch[dest] = index;
                left += lower; right += !lower;
            }
            std::memcpy(secondary.data() + lo, scratch.data() + lo, (hi - lo) * sizeof(std::uint32_t));
        };
        // The primary median defines exactly nth_element's two sets; stable partition preserves the other axis order.
        if (by_x) { split(a_x, a_y, ar_x); split(s_x, s_y, sr_x); }
        else { split(a_y, a_x, ar_y); split(s_y, s_x, sr_y); }
        self(self, lo, mid); self(self, mid, hi);
    };
    visit(visit, 0, n);
}

}  // namespace lumen::formation
