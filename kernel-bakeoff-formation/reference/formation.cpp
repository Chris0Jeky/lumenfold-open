// lumen::formation (see formation.h).
#include "formation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace lumen::formation {

namespace {

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

void assign(std::span<const float> ax, std::span<const float> ay, std::span<const float> sx, std::span<const float> sy,
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

}  // namespace lumen::formation
