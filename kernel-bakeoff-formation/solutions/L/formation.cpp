// lumen::formation (see formation.h).
#include "formation.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace lumen::formation {

namespace {

bool default_float_mode() {
#if defined(_M_X64) || defined(__SSE2__)
    // Radix keys assume ordinary IEEE comparisons, and the coarse counts were
    // computed at round-to-nearest. Honor callers using other FP modes.
    return (_mm_getcsr() & 0xffc0U) == 0x1f80U; // default controls; status flags may vary
#else
    return true;
#endif
}

// Keep the original expression and control flow for non-finite transforms:
// NaN payload selection can depend on the compiler's register allocation.
namespace original_shape {
bool ellipse(double x, double y, double cx, double cy, double rx, double ry) {
    const double dx = (x - cx) / rx, dy = (y - cy) / ry;
    return dx * dx + dy * dy <= 1.0;
}
bool triangle(double x, double y, double ax, double ay, double bx, double by, double cx, double cy) {
    auto cr = [&](double ux, double uy, double vx, double vy) { return (vx - ux) * (y - uy) - (vy - uy) * (x - ux); };
    const double a = cr(ax, ay, bx, by), b = cr(bx, by, cx, cy), c = cr(cx, cy, ax, ay);
    return (a >= 0 && b >= 0 && c >= 0) || (a <= 0 && b <= 0 && c <= 0);
}
// The silhouettes (original outlines), in [-960, 960] units.
int shape_id(std::string_view s) { return s == "disk" ? 0 : s == "sword" ? 1 : s == "hand" ? 2 : s == "dragon" ? 3 : -1; }
bool inside(int id, double x, double y) {
    auto rect = [&](double x0, double y0, double x1, double y1) { return x >= x0 && x <= x1 && y >= y0 && y <= y1; };
    switch (id) {
        case 0: return ellipse(x, y, 0, 0, 820, 820);
        case 1: return rect(-100, -330, 100, 700) || triangle(x, y, -100, 700, 100, 700, 0, 960) || rect(-440, -420, 440, -330) ||
                       rect(-60, -800, 60, -420) || ellipse(x, y, 0, -835, 115, 85);
        case 2: return rect(-360, -450, 330, 230) || ellipse(x, y, 0, -450, 360, 240) || ellipse(x, y, -280, 380, 90, 380) ||
                       ellipse(x, y, -85, 480, 90, 440) || ellipse(x, y, 110, 460, 90, 420) || ellipse(x, y, 300, 340, 85, 330) ||
                       triangle(x, y, -350, -230, -700, 170, -500, 320) || ellipse(x, y, -570, 180, 125, 140);
        case 3: return ellipse(x, y, 100, -100, 280, 460) || ellipse(x, y, 300, 430, 180, 240) || rect(340, 430, 660, 530) ||
                       triangle(x, y, 270, 600, 340, 850, 420, 570) || triangle(x, y, -100, 100, -900, 740, -700, 70) ||
                       triangle(x, y, -100, 100, -700, 70, -590, -280) || triangle(x, y, 120, 150, 850, 790, 730, 110) ||
                       triangle(x, y, 120, 150, 730, 110, 650, -250) || triangle(x, y, 50, -350, -820, -780, -230, -500) ||
                       triangle(x, y, -820, -780, -900, -510, -650, -660) || triangle(x, y, 80, -380, -40, -700, 250, -620);
    }
    return false;
}
std::size_t count(int id, double u) {
    std::size_t c = 0;
    const int k = int(std::floor(960.0 / u));
    for (int j = -k; j <= k; ++j)
        for (int i = -k; i <= k; ++i) c += inside(id, i * u, j * u);
    return c;
}

bool shape_slots(std::string_view shape, std::size_t n, float spacing, float cx, float cy, float cos_yaw, float sin_yaw,
                 std::vector<float>& sx, std::vector<float>& sy) {
    sx.clear(); sy.clear();
    const int id = shape_id(shape);
    if (id < 0 || n == 0 || !(std::isfinite(spacing) && spacing > 0.f)) return false;
    // A grid step u (units) with at least n slots inside: estimate from the fill at a coarse step, then shrink 1% at a time.
    const double fill = double(count(id, 16.0)) * 256.0;  // area in square units
    double u = std::sqrt(fill / double(n)) * 1.02;
    std::size_t m = 0;
    for (int tries = 0; tries < 400 && u >= 0.25; ++tries, u *= 0.99)
        if ((m = count(id, u)) >= n) break;
    if (m < n) return false;
    // Keep n of the m inside points, spread evenly in row-major order, as world positions.
    const double s = double(spacing) / u;  // cm per unit
    const int k = int(std::floor(960.0 / u));
#ifdef __EMSCRIPTEN__
    std::uint64_t seen = 0;  // wasm32: seen*n exceeds size_t above about 65k slots; retain the native 64-bit arithmetic
#else
    std::size_t seen = 0;
#endif
    sx.reserve(n); sy.reserve(n);
    for (int j = -k; j <= k; ++j)
        for (int i = -k; i <= k; ++i) {
            const double x = i * u, y = j * u;
            if (!inside(id, x, y)) continue;
            if ((seen + 1) * n / m > seen * n / m) {
                const double wx = s * x, wy = s * y;  // shape +y along (cos_yaw, sin_yaw), +x to its right
                sx.push_back(float(cx + wy * cos_yaw + wx * sin_yaw));
                sy.push_back(float(cy + wy * sin_yaw - wx * cos_yaw));
            }
            ++seen;
        }
    return sx.size() == n;
}

} // namespace original_shape

// Each primitive cuts a row into one interval. Search using the original
// rounded predicates, rather than trusting rounded analytic roots, so grid
// points on an edge receive exactly the same decision.
struct Primitive {
    int kind, sign; // ellipse, rectangle, triangle; triangle winding
    double a, b, c, d, e, f, ymin, ymax;
    constexpr Primitive(int kind_, double a_, double b_, double c_, double d_, double e_ = 0, double f_ = 0)
        : kind(kind_), sign((c_ - a_) * (f_ - b_) - (d_ - b_) * (e_ - a_) > 0 ? 1 : -1),
          a(a_), b(b_), c(c_), d(d_), e(e_), f(f_),
          ymin(kind_ == 0 ? b_ - d_ - 1 : kind_ == 1 ? b_ : std::min({b_, d_, f_}) - 1),
          ymax(kind_ == 0 ? b_ + d_ + 1 : kind_ == 1 ? d_ : std::max({b_, d_, f_}) + 1) {}
};
constexpr Primitive disk[] = {{0, 0, 0, 820, 820}};
constexpr Primitive sword[] = {
    {1, -100, -330, 100, 700}, {2, -100, 700, 100, 700, 0, 960},
    {1, -440, -420, 440, -330}, {1, -60, -800, 60, -420}, {0, 0, -835, 115, 85}};
constexpr Primitive hand[] = {
    {1, -360, -450, 330, 230}, {0, 0, -450, 360, 240}, {0, -280, 380, 90, 380},
    {0, -85, 480, 90, 440}, {0, 110, 460, 90, 420}, {0, 300, 340, 85, 330},
    {2, -350, -230, -700, 170, -500, 320}, {0, -570, 180, 125, 140}};
constexpr Primitive dragon[] = {
    {0, 100, -100, 280, 460}, {0, 300, 430, 180, 240}, {1, 340, 430, 660, 530},
    {2, 270, 600, 340, 850, 420, 570}, {2, -100, 100, -900, 740, -700, 70},
    {2, -100, 100, -700, 70, -590, -280}, {2, 120, 150, 850, 790, 730, 110},
    {2, 120, 150, 730, 110, 650, -250}, {2, 50, -350, -820, -780, -230, -500},
    {2, -820, -780, -900, -510, -650, -660}, {2, 80, -380, -40, -700, 250, -620}};
int shape_id(std::string_view s) { return s == "disk" ? 0 : s == "sword" ? 1 : s == "hand" ? 2 : s == "dragon" ? 3 : -1; }
std::span<const Primitive> primitives(int id) {
    switch (id) {
        case 0: return disk;
        case 1: return sword;
        case 2: return hand;
        default: return dragon;
    }
}
struct Interval { int lo, hi; };
struct Row {
    int j, size = 0;
    Interval intervals[11]; // maximum primitive count among the fixed silhouettes
};
// An analytic root only seeds the search. Adjacent probes normally settle the
// boundary; binary search remains the fallback, and every decision uses pred.
template<class Predicate>
void clip_near(int& lo, int& hi, Predicate pred, bool true_on_left, double grid_root) {
    if (lo > hi) return;
    int left = lo - 1, right = hi + 1;
    int probe = int(std::clamp(std::floor(grid_root), double(lo), double(hi)));
    const bool first = pred(probe) == true_on_left;
    if (first) left = probe; else right = probe;
    if (right - left > 1) {
        probe += first ? 1 : -1;
        if (pred(probe) == true_on_left) left = probe; else right = probe;
    }
    while (right - left > 1) {
        probe = left + (right - left) / 2;
        if (pred(probe) == true_on_left) left = probe; else right = probe;
    }
    if (true_on_left) hi = left; else lo = right;
}
Row row_intervals(int id, double u, int k, int j) {
    Row row; row.j = j;
    const double y = j * u, inv_u = 1.0 / u;
    auto add = [&](int lo, int hi) {
        if (lo <= hi) row.intervals[row.size++] = {lo, hi};
    };
    for (const auto& p : primitives(id)) {
        // A one-unit margin dwarfs possible rounded-edge displacement for
        // these bounded fixed primitives; all boundary decisions still use pred.
        if (y < p.ymin || y > p.ymax) continue;
        if (p.kind == 0) {
            auto inside = [&](int i) {
                const double dx = (i * u - p.a) / p.c, dy = (y - p.b) / p.d;
                return dx * dx + dy * dy <= 1.0;
            };
            // The nearest grid point on either side of the centre contains the
            // minimum of dx*dx. Each half is monotone under IEEE arithmetic.
            int centre = std::clamp(int(std::floor(p.a / u)), -k, k);
            if (!inside(centre)) {
                if (centre == k || !inside(centre + 1)) continue;
                ++centre;
            }
            const double dy = (y - p.b) / p.d;
            const double radius = p.c * std::sqrt(std::max(0.0, 1.0 - dy * dy));
            int lo = -k, hi = centre;
            clip_near(lo, hi, inside, false, (p.a - radius) * inv_u);
            const int first = lo;
            lo = centre; hi = k;
            clip_near(lo, hi, inside, true, (p.a + radius) * inv_u);
            add(first, hi);
        } else if (p.kind == 1) {
            if (y < p.b || y > p.d) continue;
            int lo = -k, hi = k;
            clip_near(lo, hi, [&](int i) { return i * u >= p.a; }, false, p.a * inv_u);
            clip_near(lo, hi, [&](int i) { return i * u <= p.c; }, true, p.c * inv_u);
            add(lo, hi);
        } else {
            // The fixed integer triangles have |twice-area| >= 10100. For
            // |x|,|y| <= 961 the summed cross-product error is < 1e-8, so the
            // opposite-winding branch cannot accept a point, even at an edge.
            const int sign = p.sign;
            int lo = -k, hi = k;
            auto edge = [&](double ux, double uy, double vx, double vy) {
                if (lo > hi) return;
                const double base = (vx - ux) * (y - uy), slope = vy - uy;
                auto pred = [&](int i) {
                    const double cross = base - slope * (i * u - ux);
                    return sign == 1 ? cross >= 0 : cross <= 0;
                };
                if (slope == 0) { if (lo <= hi && !pred(lo)) hi = lo - 1; }
                else clip_near(lo, hi, pred, (slope > 0) == (sign == 1), (base / slope + ux) * inv_u);
            };
            edge(p.a, p.b, p.c, p.d);
            edge(p.c, p.d, p.e, p.f);
            edge(p.e, p.f, p.a, p.b);
            add(lo, hi);
        }
    }
    std::sort(row.intervals, row.intervals + row.size,
              [](Interval a, Interval b) { return a.lo < b.lo; });
    int merged = 0;
    for (int t = 0; t < row.size; ++t) {
        const auto r = row.intervals[t];
        if (merged && r.lo <= row.intervals[merged - 1].hi + 1)
            row.intervals[merged - 1].hi = std::max(row.intervals[merged - 1].hi, r.hi);
        else row.intervals[merged++] = r;
    }
    row.size = merged;
    return row;
}
std::size_t count(int id, double u, std::vector<Row>* rows = nullptr) {
    std::size_t c = 0;
    const int k = int(std::floor(960.0 / u));
    if (rows) { rows->clear(); rows->reserve(2 * k + 1); }
    for (int j = -k; j <= k; ++j) {
        const Row row = row_intervals(id, u, k, j);
        for (int t = 0; t < row.size; ++t) c += row.intervals[t].hi - row.intervals[t].lo + 1;
        if (rows && row.size) rows->push_back(row);
    }
    return c;
}

// Reserve-and-append preserves the original behavior when both outputs alias.
// Otherwise all n entries are known in advance, so fill their storage directly.
template<bool Append>
void emit_slots(const std::vector<Row>& rows, std::size_t n, std::size_t m, double u, double scale,
                float cx, float cy, float cos_yaw, float sin_yaw,
                std::vector<float>& sx, std::vector<float>& sy) {
    if constexpr (Append) { sx.reserve(n); sy.reserve(n); }
    else { sx.resize(n); sy.resize(n); }
    auto* dx = sx.data(); auto* dy = sy.data();
    std::size_t remainder = 0, written = 0;
    for (const Row& row : rows) {
        const double y = row.j * u, wy = scale * y;
        // These are the original left-associated expressions, hoisted per row.
        const double row_x = cx + wy * cos_yaw, row_y = cy + wy * sin_yaw;
        for (int t = 0; t < row.size; ++t)
            for (int i = row.intervals[t].lo; i <= row.intervals[t].hi; ++i) {
                remainder += n;
                if (remainder >= m) {
                    remainder -= m;
                    const double x = i * u, wx = scale * x;
                    const float ox = float(row_x + wx * sin_yaw), oy = float(row_y - wx * cos_yaw);
                    if constexpr (Append) { sx.push_back(ox); sy.push_back(oy); }
                    else { dx[written] = ox; dy[written] = oy; ++written; }
                }
            }
    }
}

}  // namespace

bool shape_slots(std::string_view shape, std::size_t n, float spacing, float cx, float cy, float cos_yaw, float sin_yaw,
                 std::vector<float>& sx, std::vector<float>& sy) {
    if (!default_float_mode() || !(std::isfinite(cx) && std::isfinite(cy) && std::isfinite(cos_yaw) && std::isfinite(sin_yaw)))
        return original_shape::shape_slots(shape, n, spacing, cx, cy, cos_yaw, sin_yaw, sx, sy);
    sx.clear(); sy.clear();
    const int id = shape_id(shape);
    if (id < 0 || n == 0 || !(std::isfinite(spacing) && spacing > 0.f)) return false;
    // A grid step u (units) with at least n slots inside: estimate from the fill at a coarse step, then shrink 1% at a time.
    // The fixed 16-unit coarse grid has these exact integer counts.
    constexpr std::size_t coarse_counts[] = {8269, 1538, 4069, 4557};
    const double fill = double(coarse_counts[id]) * 256.0;  // area in square units
    double u = std::sqrt(fill / double(n)) * 1.02;
    std::size_t m = 0;
    std::vector<Row> rows;
    for (int tries = 0; tries < 400 && u >= 0.25; ++tries, u *= 0.99)
        if ((m = count(id, u, &rows)) >= n) break;
    if (m < n) return false;
    // Keep n of the m inside points, spread evenly in row-major order, as world positions.
    const double s = double(spacing) / u;  // cm per unit
    // The merged intervals enumerate the same inside cells in row-major order.
    // n <= m <= 7681^2, so a remainder accumulator cannot overflow; it equals
    // the difference between the original consecutive floor(seen*n/m) values.
    if (&sx == &sy) emit_slots<true>(rows, n, m, u, s, cx, cy, cos_yaw, sin_yaw, sx, sy);
    else emit_slots<false>(rows, n, m, u, s, cx, cy, cos_yaw, sin_yaw, sx, sy);
    return sx.size() == n;
}

static void assign_original(std::span<const float> ax, std::span<const float> ay, std::span<const float> sx, std::span<const float> sy,
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

namespace {
struct Point {
    float x, y;
    std::uint32_t index;
};
template<bool ByX>
bool point_less(const Point& a, const Point& b) {
    const float ka = ByX ? a.x : a.y, kb = ByX ? b.x : b.y;
    return ka != kb ? ka < kb : a.index < b.index;
}
template<bool ByX>
void split_points(Point* begin, Point* mid, Point* end) {
    std::nth_element(begin, mid, end, point_less<ByX>);
}
void match_points(Point* a, Point* s, std::size_t n, std::uint32_t* out) {
    if (n == 1) { out[a[0].index] = s[0].index; return; }
    float x0 = a[0].x, x1 = x0, y0 = a[0].y, y1 = y0;
    for (std::size_t k = 0; k < n; ++k) {
        x0 = std::min({x0, a[k].x, s[k].x}); x1 = std::max({x1, a[k].x, s[k].x});
        y0 = std::min({y0, a[k].y, s[k].y}); y1 = std::max({y1, a[k].y, s[k].y});
    }
    const bool by_x = x1 - x0 >= y1 - y0;
    if (n == 2) {
        // With two points, the only split is their total-order comparison.
        const bool aa = by_x ? point_less<true>(a[0], a[1]) : point_less<false>(a[0], a[1]);
        const bool ss = by_x ? point_less<true>(s[0], s[1]) : point_less<false>(s[0], s[1]);
        out[a[0].index] = s[aa == ss ? 0 : 1].index;
        out[a[1].index] = s[aa == ss ? 1 : 0].index;
        return;
    }
    const auto mid = n / 2;
    if (by_x) { split_points<true>(a, a + mid, a + n); split_points<true>(s, s + mid, s + n); }
    else { split_points<false>(a, a + mid, a + n); split_points<false>(s, s + mid, s + n); }
    match_points(a, s, mid, out);
    match_points(a + mid, s + mid, n - mid, out);
}

std::uint32_t float_key(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    // Both zero signs compare equal in the original comparator.
    if ((bits & 0x7fffffffU) == 0) bits = 0;
    return bits & 0x80000000U ? ~bits : bits ^ 0x80000000U;
}
void radix_order(std::uint32_t* order, std::uint32_t* tmp, const std::uint32_t* keys, std::size_t n) {
    // A permutation does not change any digit's histogram. Count all digits in
    // one sequential pass, then reuse those counts for the four stable passes.
    auto* original = order;
    std::array<std::array<std::uint32_t, 256>, 4> histograms{};
    for (std::size_t i = 0; i < n; ++i) {
        order[i] = std::uint32_t(i);
        const auto key = keys[i];
        for (unsigned digit = 0; digit != 4; ++digit) ++histograms[digit][(key >> (digit * 8)) & 255];
    }
    for (unsigned digit = 0; digit != 4; ++digit) {
        auto& histogram = histograms[digit];
        std::uint32_t total = 0;
        bool uniform = false;
        for (auto& bucket : histogram) {
            const auto count = bucket;
            if (count == n) { uniform = true; break; }
            bucket = total; total += count;
        }
        // A uniform digit preserves order, so its stable pass is an identity.
        if (uniform) continue;
        for (std::size_t i = 0; i < n; ++i) {
            const auto index = order[i];
            const auto key = keys[index];
            tmp[histogram[(key >> (digit * 8)) & 255]++] = index;
        }
        std::swap(order, tmp);
    }
    if (order != original) std::copy_n(order, n, original);
}

struct Coordinates {
    unsigned char* marks;
    const float *px, *py;
};
struct Orders {
    std::uint32_t *x, *y, *tmp;
    const Coordinates* coordinates;
    Orders child(bool by_x, std::size_t offset) const {
        return by_x ? Orders{x + offset, tmp + offset, y + offset, coordinates}
                    : Orders{tmp + offset, y + offset, x + offset, coordinates};
    }

};
// Partition the two independent point sets together: this shares loop overhead
// and lets the CPU overlap their independent membership loads and counters.
void partition_orders(Orders a, Orders s, bool by_x, std::size_t n, std::size_t mid, unsigned char tag) {
    const auto am = a.coordinates->marks, sm = s.coordinates->marks;
    const auto ac = by_x ? a.x : a.y, sc = by_x ? s.x : s.y;
    for (std::size_t i = 0; i < mid; ++i) { am[ac[i]] = tag; sm[sc[i]] = tag; }
    const auto ao = by_x ? a.y : a.x, so = by_x ? s.y : s.x;
    std::size_t al = 0, sl = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto ai = ao[i], si = so[i];
        const bool low_a = am[ai] == tag, low_s = sm[si] == tag;
        const auto ad = low_a ? al : mid + i - al;
        const auto sd = low_s ? sl : mid + i - sl;
        al += low_a; sl += low_s;
        a.tmp[ad] = ai; s.tmp[sd] = si;
    }
}
// These pairs arrive sorted on the parent's axis. If their orders on the
// other axis agree as well, either child axis produces the same two matches.
void match_pair(std::uint32_t a0, std::uint32_t a1, std::uint32_t s0, std::uint32_t s1,
                const Coordinates& a, const Coordinates& s, bool parent_x, std::uint32_t* out) {
    const auto ap = parent_x ? a.py : a.px, sp = parent_x ? s.py : s.px;
    auto less = [](float x, float y, std::uint32_t i, std::uint32_t j) {
        return x != y ? x < y : i < j;
    };
    bool flip = less(ap[a0], ap[a1], a0, a1) != less(sp[s0], sp[s1], s0, s1);
    if (flip) {
        const float x0 = std::min({a.px[a0], a.px[a1], s.px[s0], s.px[s1]});
        const float x1 = std::max({a.px[a0], a.px[a1], s.px[s0], s.px[s1]});
        const float y0 = std::min({a.py[a0], a.py[a1], s.py[s0], s.py[s1]});
        const float y1 = std::max({a.py[a0], a.py[a1], s.py[s0], s.py[s1]});
        flip = (x1 - x0 >= y1 - y0) != parent_x;
    }
    out[a0] = flip ? s1 : s0;
    out[a1] = flip ? s0 : s1;
}
void match_orders(Orders a, Orders s, std::size_t n, std::uint32_t* out, unsigned char tag = 1) {
    if (n == 1) { out[a.x[0]] = s.x[0]; return; }
    if (n == 2 && ((a.x[0] == a.y[0]) == (s.x[0] == s.y[0]))) {
        out[a.x[0]] = s.x[0]; out[a.x[1]] = s.x[1]; return;
    }
    // Sorted endpoints are the exact extrema; only their zero sign may differ,
    // which cannot affect the comparison of the two nonnegative extents.
    const float x0 = std::min(a.coordinates->px[a.x[0]], s.coordinates->px[s.x[0]]);
    const float x1 = std::max(a.coordinates->px[a.x[n - 1]], s.coordinates->px[s.x[n - 1]]);
    const float y0 = std::min(a.coordinates->py[a.y[0]], s.coordinates->py[s.y[0]]);
    const float y1 = std::max(a.coordinates->py[a.y[n - 1]], s.coordinates->py[s.y[n - 1]]);
    const bool by_x = x1 - x0 >= y1 - y0;
    if (n == 2) {
        const auto aa = by_x ? a.x : a.y, ss = by_x ? s.x : s.y;
        out[aa[0]] = ss[0]; out[aa[1]] = ss[1]; return;
    }
    if (n <= 4) {
        const auto aa = by_x ? a.x : a.y, ss = by_x ? s.x : s.y;
        if (n == 3) out[aa[0]] = ss[0];
        else match_pair(aa[0], aa[1], ss[0], ss[1], *a.coordinates, *s.coordinates, by_x, out);
        match_pair(aa[n - 2], aa[n - 1], ss[n - 2], ss[n - 1], *a.coordinates, *s.coordinates, by_x, out);
        return;
    }
    const auto mid = n / 2;
    partition_orders(a, s, by_x, n, mid, tag);
    match_orders(a.child(by_x, 0), s.child(by_x, 0), mid, out, tag + 1);
    match_orders(a.child(by_x, mid), s.child(by_x, mid), n - mid, out, tag + 1);
}
void assign_presorted(std::span<const float> ax, std::span<const float> ay,
                      std::span<const float> sx, std::span<const float> sy,
                      std::vector<std::uint32_t>& out) {
    const auto n = ax.size();
    // Reuse only scratch capacity, never results: all keys, orders and marks
    // are rebuilt on every call. TLS keeps independent callers independent.
    struct Scratch {
        std::vector<std::uint32_t> ar, sr;
        std::vector<std::uint32_t> storage;
    };
    thread_local Scratch scratch;
    scratch.ar.resize(2 * n); scratch.sr.resize(2 * n); scratch.storage.resize(6 * n);
    auto& ar = scratch.ar; auto& sr = scratch.sr;
    auto* a = scratch.storage.data(); auto* s = a + 3 * n;
    bool finite = true;
    std::size_t i = 0;
#if defined(_M_X64) || defined(__SSE2__)
    // SSE2 is part of every x64 CPU. These are integer operations on the float
    // bits, so no floating-point arithmetic or rounding is changed.
    const auto sign = _mm_set1_epi32(std::numeric_limits<std::int32_t>::min());
    const auto exponent = _mm_set1_epi32(0x7f800000);
    auto invalid = _mm_setzero_si128();
    auto keys = [&](const float* values, std::uint32_t* dest) {
        auto v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
        invalid = _mm_or_si128(invalid, _mm_cmpeq_epi32(_mm_and_si128(v, exponent), exponent));
        v = _mm_andnot_si128(_mm_cmpeq_epi32(v, sign), v); // canonicalize -0
        v = _mm_xor_si128(v, _mm_or_si128(_mm_srai_epi32(v, 31), sign));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dest), v);
    };
    for (; i + 4 <= n; i += 4) {
        keys(ax.data() + i, ar.data() + i); keys(ay.data() + i, ar.data() + n + i);
        keys(sx.data() + i, sr.data() + i); keys(sy.data() + i, sr.data() + n + i);
    }
    finite = _mm_movemask_epi8(invalid) == 0;
#endif
    for (; i < n; ++i) {
        finite = finite && std::isfinite(ax[i]) && std::isfinite(ay[i]) && std::isfinite(sx[i]) && std::isfinite(sy[i]);
        ar[i] = float_key(ax[i]); ar[n + i] = float_key(ay[i]);
        sr[i] = float_key(sx[i]); sr[n + i] = float_key(sy[i]);
    }
    if (!finite) { assign_original(ax, ay, sx, sy, out); return; }
    // Stable radix sorts start in index order: (coordinate,index) is therefore
    // exactly the original total order, including ties and signed zero.
    radix_order(a, a + 2 * n, ar.data(), n);
    radix_order(a + n, a + 2 * n, ar.data() + n, n);
    radix_order(s, s + 2 * n, sr.data(), n);
    radix_order(s + n, s + 2 * n, sr.data() + n, n);
    // A depth tag marks the chosen half. Ancestors have smaller tags and
    // sibling subtrees have disjoint indices, so no clearing is needed below
    // the root. At most 32 levels are possible for uint32_t indices.
    auto* am = reinterpret_cast<unsigned char*>(ar.data());
    auto* sm = reinterpret_cast<unsigned char*>(sr.data());
    std::fill_n(am, n, static_cast<unsigned char>(0)); std::fill_n(sm, n, static_cast<unsigned char>(0));
    out.resize(n);
    // Restrict the other sorted order to each rank half. Ping-ponging its
    // buffer preserves both orders without rescans for bounds or copies.
    const Coordinates ac{am, ax.data(), ay.data()}, sc{sm, sx.data(), sy.data()};
    match_orders({a, a + n, a + 2 * n, &ac},
                 {s, s + n, s + 2 * n, &sc}, n, out.data());
}
} // namespace

void assign(std::span<const float> ax, std::span<const float> ay, std::span<const float> sx, std::span<const float> sy,
            std::vector<std::uint32_t>& slot_of) {
    const auto n = ax.size();
    if (ay.size() != n || sx.size() != n || sy.size() != n)
        throw std::invalid_argument("lumen::formation::assign: one length for agents and slots");
    // Numeric coordinates plus the index form a unique total order. Thus each
    // rank split selects the same sets regardless of nth_element's arrangement.
    // Keep the original path for unusual FP modes, wrapped indices and NaNs.
    if (!default_float_mode() || n > std::numeric_limits<std::uint32_t>::max()) {
        assign_original(ax, ay, sx, sy, slot_of); return;
    }
    if (n >= 512) { assign_presorted(ax, ay, sx, sy, slot_of); return; }
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(ax[i]) || !std::isfinite(ay[i]) || !std::isfinite(sx[i]) || !std::isfinite(sy[i])) {
            assign_original(ax, ay, sx, sy, slot_of);
            return;
        }
    std::vector<Point> a(n), s(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = {ax[i], ay[i], std::uint32_t(i)};
        s[i] = {sx[i], sy[i], std::uint32_t(i)};
    }
    slot_of.assign(n, 0);
    if (n) match_points(a.data(), s.data(), n, slot_of.data());
}

}  // namespace lumen::formation
