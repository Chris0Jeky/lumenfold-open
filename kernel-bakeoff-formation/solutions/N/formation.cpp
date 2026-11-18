// lumen::formation (see formation.h).
#include "formation.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace lumen::formation {

namespace {

int shape_id(std::string_view s) { return s == "disk" ? 0 : s == "sword" ? 1 : s == "hand" ? 2 : s == "dragon" ? 3 : -1; }

struct Interval {
    int first, end;
};

// The estimate only saves probes. The two adjacent grid points certify the
// boundary of a monotone predicate, with binary search if the estimate is off.
template<class Predicate>
int first_true(int first, int end, int estimate, Predicate test) {
    int point = std::clamp(estimate, first, end);
    if (point < end && !test(point)) {
        ++point;
        if (point == end || test(point)) return point;
        first = point + 1;
    } else {
        if (point == first || !test(point - 1)) return point;
        end = point - 1;
    }
    while (first < end) {
        const int mid = first + (end - first) / 2;
        if (test(mid)) end = mid;
        else first = mid + 1;
    }
    return first;
}

struct Edge {
    double x, y, dx, dy, inverse_dy;
    constexpr Edge(double ux, double uy, double vx, double vy)
        : x(ux), y(uy), dx(vx - ux), dy(vy - uy), inverse_dy(dy != 0.0 ? 1.0 / dy : 0.0) {}
};

struct Triangle {
    Edge edges[3];
    double sign;

    constexpr Triangle(double ax, double ay, double bx, double by, double cx, double cy)
        : edges{{ax, ay, bx, by}, {bx, by, cx, cy}, {cx, cy, ax, ay}},
          sign((bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0 ? 1 : -1) {}
};

struct Grid {
    double u, inverse;
    int k;

    explicit Grid(double step) : u(step), inverse(1.0 / step), k(int(std::floor(960.0 / step))) {}
    double coordinate(int i) const { return i * u; }
    int estimate(double x) const {
        return int(std::clamp(x * inverse, double(-k - 1), double(k + 1)));
    }

    // On this bounded binary64 grid, rounding shifts a triangle cut by
    // less than 2^-20 columns and even an ellipse tip by less than 2^-10
    // (see README). Away from a grid point, either inclusion rule has the
    // same cut. Near an integer the caller uses the original predicate.
    bool unambiguous_cut(double crossing, int& cut) const {
        constexpr double margin = 1.0 / 512.0;
        const double pos = crossing * inverse;
        if (pos < -k - margin) {
            cut = -k;
            return true;
        }
        if (pos > k + margin) {
            cut = k + 1;
            return true;
        }
        constexpr int bias = 4096;  // the grid reaches at most 3840 in either direction
        const double shifted = pos + bias;
        const int lower = int(shifted);
        const double fraction = shifted - lower;
        cut = lower - bias + 1;
        return fraction > margin && fraction < 1.0 - margin;
    }
};

// The silhouettes (original outlines). A row is the union
// of a few convex intervals. Use the original arithmetic at their boundaries.
struct Row {
    const Grid& grid;
    double y;
    Interval intervals[11];  // at most one per primitive, eleven in dragon
    int size = 0;

    Row(const Grid& g, int j) : grid(g), y(g.coordinate(j)) {}

    void add(int first, int end) {
        if (first >= end) return;
        int i = size++;
        while (i > 0 && intervals[i - 1].first > first) {
            intervals[i] = intervals[i - 1];
            --i;
        }
        intervals[i] = {first, end};
    }

    void rect(double x0, double y0, double x1, double y1) {
        if (y < y0 || y > y1) return;
        const int first = first_true(-grid.k, grid.k + 1, grid.estimate(x0),
                                    [&](int i) { return grid.coordinate(i) >= x0; });
        const int end = first_true(first, grid.k + 1, grid.estimate(x1) + 1,
                                  [&](int i) { return grid.coordinate(i) > x1; });
        add(first, end);
    }

    void ellipse(double cx, double cy, double rx, double ry) {
        const double dy = (y - cy) / ry;
        const double dy2 = dy * dy;
        if (dy2 > 1.0) return;
        const double reach = rx * std::sqrt(1.0 - dy2);
        int first_guess, end_guess;
        if (grid.unambiguous_cut(cx - reach, first_guess) &&
            grid.unambiguous_cut(cx + reach, end_guess)) {
            add(first_guess, end_guess);
            return;
        }
        const int center = first_true(-grid.k, grid.k + 1, grid.estimate(cx),
                                     [&](int i) { return grid.coordinate(i) >= cx; });
        // Squaring is monotone on either side of the center. Test the sum
        // as written in the original, including rounding at the boundary.
        auto inside = [&](int i) {
            const double dx = (grid.coordinate(i) - cx) / rx;
            return dx * dx + dy2 <= 1.0;
        };
        const int first = first_true(-grid.k, center, grid.estimate(cx - reach), inside);
        const int end = first_true(center, grid.k + 1, grid.estimate(cx + reach) + 1,
                                  [&](int i) { return !inside(i); });
        add(first, end);
    }

    void triangle(const Triangle& t) {
        int first = -grid.k;
        int end = grid.k + 1;
        // The signed cross products sum to the fixed doubled area. These
        // integer-vertex triangles have area far larger than the sub-unit
        // rounding error on the grid, so only their given winding can pass.
        for (const Edge& e : t.edges) {
            const double a = e.dx * (y - e.y);
            if (e.dy == 0) {
                if (a * t.sign < 0) return;
                continue;
            }
            auto inside = [&](int i) {
                return (a - e.dy * (grid.coordinate(i) - e.x)) * t.sign >= 0;
            };
            const double crossing = e.x + a * e.inverse_dy;
            const bool lower = e.dy * t.sign < 0;
            int cut;
            if (grid.unambiguous_cut(crossing, cut)) {
                if (lower) first = std::clamp(cut, first, end);
                else end = std::clamp(cut, first, end);
            } else {
                const int boundary = grid.estimate(crossing);
                if (lower) first = first_true(first, end, boundary, inside);
                else end = first_true(first, end, boundary + 1, [&](int i) { return !inside(i); });
            }
            if (first == end) return;
        }
        add(first, end);
    }

    void silhouette(int id) {
        switch (id) {
            case 0:
                ellipse(0, 0, 820, 820);
                break;
            case 1:
                rect(-100, -330, 100, 700);
                triangle({-100, 700, 100, 700, 0, 960});
                rect(-440, -420, 440, -330);
                rect(-60, -800, 60, -420);
                ellipse(0, -835, 115, 85);
                break;
            case 2:
                rect(-360, -450, 330, 230);
                ellipse(0, -450, 360, 240);
                ellipse(-280, 380, 90, 380);
                ellipse(-85, 480, 90, 440);
                ellipse(110, 460, 90, 420);
                ellipse(300, 340, 85, 330);
                triangle({-350, -230, -700, 170, -500, 320});
                ellipse(-570, 180, 125, 140);
                break;
            case 3:
                ellipse(100, -100, 280, 460);
                ellipse(300, 430, 180, 240);
                rect(340, 430, 660, 530);
                triangle({270, 600, 340, 850, 420, 570});
                triangle({-100, 100, -900, 740, -700, 70});
                triangle({-100, 100, -700, 70, -590, -280});
                triangle({120, 150, 850, 790, 730, 110});
                triangle({120, 150, 730, 110, 650, -250});
                triangle({50, -350, -820, -780, -230, -500});
                triangle({-820, -780, -900, -510, -650, -660});
                triangle({80, -380, -40, -700, 250, -620});
                break;
        }
    }

    template<class Visit>
    void visit(Visit emit) const {
        if (size == 0) return;
        Interval merged = intervals[0];
        for (int i = 1; i < size; ++i) {
            const Interval next = intervals[i];
            if (next.first <= merged.end) merged.end = std::max(merged.end, next.end);
            else {
                emit(merged);
                merged = next;
            }
        }
        emit(merged);
    }
};

struct GridRun {
    int row;
    Interval columns;
};

template<bool Save = false>
std::size_t count(int id, double u, std::vector<GridRun>* saved = nullptr) {
    const Grid grid(u);
    std::size_t total = 0;
    if constexpr (Save) {
        saved->clear();
        if (saved->capacity() == 0) saved->reserve(5 * std::size_t(grid.k) + 4);
    }
    for (int j = -grid.k; j <= grid.k; ++j) {
        Row row(grid, j);
        row.silhouette(id);
        row.visit([&](Interval run) {
            total += std::size_t(run.end - run.first);
            if constexpr (Save) saved->push_back({j, run});
        });
    }
    return total;
}

}  // namespace

bool shape_slots(std::string_view shape, std::size_t n, float spacing, float cx, float cy, float cos_yaw, float sin_yaw,
                 std::vector<float>& sx, std::vector<float>& sy) {
    sx.clear();
    sy.clear();
    const int id = shape_id(shape);
    if (id < 0 || n == 0 || !(std::isfinite(spacing) && spacing > 0.f)) return false;
    // A grid step u (units) with at least n slots inside: estimate from the fill at a coarse step, then shrink 1% at a time.
    const double fill = double(count(id, 16.0)) * 256.0;  // area in square units
    double u = std::sqrt(fill / double(n)) * 1.02;
    std::vector<GridRun> saved;
    std::size_t m = 0;
    for (int tries = 0; tries < 400 && u >= 0.25; ++tries, u *= 0.99)
        if ((m = count<true>(id, u, &saved)) >= n) break;
    if (m < n) return false;
    // Keep n of the m inside points, spread evenly in row-major order, as world positions.
    const double s = double(spacing) / u;  // cm per unit
    const Grid grid(u);
    // Carry is (seen * n) % m, so crossing m is the original floor-division
    // test. With m <= 7681^2 the original products fit in 64 bits.
    sx.reserve(n);
    sy.reserve(n);
    // Separate outputs can be sized once. Keep the original interleaved
    // pushes for aliased vectors and scalar expressions for nonfinite values.
    if (&sx != &sy && std::isfinite(cx) && std::isfinite(cy) &&
        std::isfinite(cos_yaw) && std::isfinite(sin_yaw)) {
        sx.resize(n);
        sy.resize(n);
        float* out_x = sx.data();
        float* out_y = sy.data();
        std::size_t carry = 0;
        for (const auto& [row, run] : saved) {
            const double wy = s * grid.coordinate(row);
            const double row_x = cx + wy * cos_yaw, row_y = cy + wy * sin_yaw;
            for (int i = run.first; i < run.end; ++i) {
                carry += n;
                if (carry < m) continue;
                carry -= m;
                const double wx = s * grid.coordinate(i);
                *out_x++ = float(row_x + wx * sin_yaw);
                *out_y++ = float(row_y - wx * cos_yaw);
            }
        }
        return true;
    }
    std::size_t carry = 0;
    for (const auto& [row, run] : saved) {
        const double wy = s * grid.coordinate(row);
        for (int i = run.first; i < run.end; ++i) {
            carry += n;
            if (carry < m) continue;
            carry -= m;
            const double wx = s * grid.coordinate(i);
            sx.push_back(float(cx + wy * cos_yaw + wx * sin_yaw));
            sy.push_back(float(cy + wy * sin_yaw - wx * cos_yaw));
        }
    }
    return sx.size() == n;
}

namespace {

// Unordered coordinates keep the original nth_element path, including its
// traversal order. Subnormals also use it, even if the FP unit flushes them.
void assign_original(std::span<const float> ax, std::span<const float> ay,
                     std::span<const float> sx, std::span<const float> sy,
                     std::vector<std::uint32_t>& slot_of) {
    const std::size_t n = ax.size();
    std::vector<std::uint32_t> A(n), S(n);
    for (std::size_t i = 0; i < n; ++i) { A[i] = std::uint32_t(i); S[i] = std::uint32_t(i); }
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

// Three 21-bit fields keep the input index beside both ranks in one word.
struct Ranks {
    static constexpr unsigned field_bits = 21;
    static constexpr std::uint32_t limit = 1u << field_bits;
    static constexpr std::uint64_t mask = limit - 1;
    std::uint64_t entry;

    // During rank construction, the output and spare arrays temporarily hold
    // floating sort keys and their input indices in the same-sized records.
    static Ranks key_index(std::uint32_t key, std::uint32_t index) {
        return {std::uint64_t(key) << 32 | index};
    }
    std::uint32_t key() const { return std::uint32_t(entry >> 32); }
    std::uint32_t key_input_index() const { return std::uint32_t(entry); }

    static Ranks make(std::uint32_t x, std::uint32_t y, std::uint32_t index) {
        return {(std::uint64_t(y) << (2 * field_bits)) |
                (std::uint64_t(index) << field_bits) | x};
    }
    std::uint32_t x() const { return std::uint32_t(entry & mask); }
    std::uint32_t y() const { return std::uint32_t(entry >> (2 * field_bits)); }
    std::uint32_t index() const { return std::uint32_t((entry >> field_bits) & mask); }
};

template<unsigned Shift>
void radix_pass(Ranks*& keys, Ranks*& spare, std::size_t n, std::uint32_t (&offsets)[256],
                std::uint32_t varied) {
    if (((varied >> Shift) & 255u) == 0) return;
    std::uint32_t pos = 0;
    for (auto& count : offsets) {
        const std::uint32_t next = pos + count;
        count = pos;
        pos = next;
    }
    for (std::size_t i = 0; i < n; ++i) {
        const Ranks point = keys[i];
        spare[offsets[(point.key() >> Shift) & 255u]++] = point;
    }
    std::swap(keys, spare);
}

bool make_order_key(float coordinate, std::uint32_t& key) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(coordinate);
    const std::uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u || (magnitude != 0 && magnitude < 0x00800000u)) return false;
    key = magnitude == 0 ? 0x80000000u : bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
    return true;
}

bool write_sorted_keys(std::span<const float> coords, Ranks* keys) {
    std::uint32_t previous = 0;
    for (std::size_t i = 0; i < coords.size(); ++i) {
        std::uint32_t key;
        if (!make_order_key(coords[i], key) || key < previous) return false;
        keys[i] = Ranks::key_index(key, std::uint32_t(i));
        previous = key;
    }
    return true;
}

// Stable radix passes preserve index order for equal coordinates. The key
// transform maps IEEE floats into ascending unsigned order and joins +/-0.
bool radix_order(std::span<const float> coords, Ranks* keys, Ranks* spare, Ranks*& sorted) {
    if (write_sorted_keys(coords, keys)) {
        sorted = keys;
        return true;
    }
    std::uint32_t histogram[4][256]{};
    std::uint32_t any_bits = 0;
    std::uint32_t all_bits = ~std::uint32_t{0};
    const std::size_t n = coords.size();
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t key;
        if (!make_order_key(coords[i], key)) return false;
        keys[i] = Ranks::key_index(key, std::uint32_t(i));
        any_bits |= key;
        all_bits &= key;
        ++histogram[0][key & 255u];
        ++histogram[1][(key >> 8) & 255u];
        ++histogram[2][(key >> 16) & 255u];
        ++histogram[3][key >> 24];
    }
    const std::uint32_t varied = any_bits ^ all_bits;
    radix_pass<0>(keys, spare, n, histogram[0], varied);
    radix_pass<8>(keys, spare, n, histogram[1], varied);
    radix_pass<16>(keys, spare, n, histogram[2], varied);
    radix_pass<24>(keys, spare, n, histogram[3], varied);
    sorted = keys;
    return true;
}

float coordinate_from_key(std::uint32_t key) {
    const std::uint32_t bits = key ^ (key >> 31 ? 0x80000000u : 0xffffffffu);
    return std::bit_cast<float>(bits);
}

struct Orders {
    Ranks* x;
    Ranks* y;
    Ranks* spare;

    Orders after(std::size_t offset) const {
        return {x + offset, y + offset, spare + offset};
    }
};

struct RankSet {
    std::unique_ptr<Ranks[]> order;
    std::unique_ptr<float[]> coordinates;

    explicit RankSet(std::size_t n)
        : order(std::make_unique_for_overwrite<Ranks[]>(3 * n)),
          coordinates(std::make_unique_for_overwrite<float[]>(2 * n)) {}

    Orders orders(std::size_t n) {
        Ranks* p = order.get();
        return {p, p + n, p + 2 * n};
    }

    bool build(std::span<const float> x, std::span<const float> y, std::uint32_t* x_rank) {
        const std::size_t n = x.size();
        const Orders o = orders(n);
        Ranks* sorted;
        if (!radix_order(x, o.y, o.spare, sorted)) return false;
        for (std::size_t rank = 0; rank < n; ++rank) {
            const std::uint32_t i = sorted[rank].key_input_index();
            x_rank[i] = std::uint32_t(rank);
            coordinates[rank] = coordinate_from_key(sorted[rank].key());
        }
        if (!radix_order(y, o.y, o.spare, sorted)) return false;
        float* const y_coords = coordinates.get() + n;
        for (std::size_t rank = 0; rank < n; ++rank) {
            const std::uint32_t i = sorted[rank].key_input_index();
            const Ranks point = Ranks::make(x_rank[i], std::uint32_t(rank), i);
            // Read the old key before replacing it when sorted == o.y.
            y_coords[rank] = coordinate_from_key(sorted[rank].key());
            o.x[point.x()] = o.y[rank] = point;
        }
        return true;
    }
};

template<bool ByX>
void split_other_orders(Orders& agents, Orders& slots, std::size_t left, std::size_t count) {
    auto rank = [](Ranks point) { return ByX ? point.x() : point.y(); };
    const auto a_mid = rank((ByX ? agents.x : agents.y)[left]);
    const auto s_mid = rank((ByX ? slots.x : slots.y)[left]);
    Ranks*& a_other = ByX ? agents.y : agents.x;
    Ranks*& s_other = ByX ? slots.y : slots.x;
    // The unchanged primary order keeps its slices. The old other order
    // becomes scratch, so the same third buffer works for either axis.
    Ranks*& a_spare = agents.spare;
    Ranks*& s_spare = slots.spare;
    Ranks* a_write[] = {a_spare + left, a_spare};
    Ranks* s_write[] = {s_spare + left, s_spare};
    for (std::size_t i = 0; i < count; ++i) {
        const Ranks a = a_other[i], s = s_other[i];
        // Advance only the chosen cursor. Each half retains its input order.
        *(a_write[rank(a) < a_mid]++) = a;
        *(s_write[rank(s) < s_mid]++) = s;
    }
    std::swap(a_other, a_spare);
    std::swap(s_other, s_spare);
}

struct Assignment {
    const float *ax, *ay, *sx, *sy;
    std::uint32_t* output;

    void match(Ranks agent, Ranks slot) const { output[agent.index()] = slot.index(); }


    // For three or four points the first split leaves at most two on each
    // side. Recompute each pair's box without writing any temporary orders.
    template<bool ParentX>
    void match_pair(Ranks a0, Ranks a1, Ranks s0, Ranks s1) const {
        const bool flip_a = ParentX ? a1.y() < a0.y() : a1.x() < a0.x();
        const bool flip_s = ParentX ? s1.y() < s0.y() : s1.x() < s0.x();
        const Ranks alo = flip_a ? a1 : a0, ahi = flip_a ? a0 : a1;
        const Ranks slo = flip_s ? s1 : s0, shi = flip_s ? s0 : s1;
        const float x0 = std::min(ax[ParentX ? a0.x() : alo.x()], sx[ParentX ? s0.x() : slo.x()]);
        const float x1 = std::max(ax[ParentX ? a1.x() : ahi.x()], sx[ParentX ? s1.x() : shi.x()]);
        const float y0 = std::min(ay[ParentX ? alo.y() : a0.y()], sy[ParentX ? slo.y() : s0.y()]);
        const float y1 = std::max(ay[ParentX ? ahi.y() : a1.y()], sy[ParentX ? shi.y() : s1.y()]);
        const bool by_x = x1 - x0 >= y1 - y0;
        if (by_x != ParentX && flip_a != flip_s) std::swap(s0, s1);
        match(a0, s0);
        match(a1, s1);
    }

    template<bool ByX>
    void match_small(const Ranks* a, const Ranks* s, std::size_t count) const {
        if (count == 2) {
            match(a[0], s[0]);
            match(a[1], s[1]);
        } else if (count == 3) {
            match(a[0], s[0]);
            match_pair<ByX>(a[1], a[2], s[1], s[2]);
        } else {
            match_pair<ByX>(a[0], a[1], s[0], s[1]);
            match_pair<ByX>(a[2], a[3], s[2], s[3]);
        }
    }

    void split(Orders agents, Orders slots, std::size_t count) const {
        if (count == 1) {
            match(agents.x[0], slots.x[0]);
            return;
        }
        // A node keeps both orders. Endpoint coordinates are its extrema.
        // Signed zeros compare equal and cannot change the axis decision.
        const std::size_t last = count - 1;
        const float x0 = std::min(ax[agents.x[0].x()], sx[slots.x[0].x()]);
        const float x1 = std::max(ax[agents.x[last].x()], sx[slots.x[last].x()]);
        const float y0 = std::min(ay[agents.y[0].y()], sy[slots.y[0].y()]);
        const float y1 = std::max(ay[agents.y[last].y()], sy[slots.y[last].y()]);
        const bool by_x = x1 - x0 >= y1 - y0;
        if (count <= 4) {
            if (by_x) match_small<true>(agents.x, slots.x, count);
            else match_small<false>(agents.y, slots.y, count);
            return;
        }
        // The median rank uniquely selects the same two sets as nth_element.
        // Stable partition of the other axis preserves its order in both sets.
        const std::size_t left = count / 2;
        if (by_x) {
            split_other_orders<true>(agents, slots, left, count);
        } else {
            split_other_orders<false>(agents, slots, left, count);
        }
        split(agents, slots, left);
        split(agents.after(left), slots.after(left), count - left);
    }
};

}  // namespace

void assign(std::span<const float> ax, std::span<const float> ay, std::span<const float> sx, std::span<const float> sy,
            std::vector<std::uint32_t>& slot_of) {
    const std::size_t n = ax.size();
    if (ay.size() != n || sx.size() != n || sy.size() != n)
        throw std::invalid_argument("lumen::formation::assign: one length for agents and slots");

    if (n < 2) {
        slot_of.assign(n, 0);
        return;
    }
    // Short inputs cost less to select directly. The packed path also needs
    // both ranks and both input indices to fit in their 21-bit fields.
    if (n >= 32 && n <= Ranks::limit) {
        RankSet agents(n), slots(n);
        // Construction uses the output as its temporary index-to-rank map.
        // Matching overwrites every entry once, or the fallback resets it.
        if (slot_of.size() != n) slot_of.assign(n, 0);
        auto* const x_rank = slot_of.data();
        if (agents.build(ax, ay, x_rank) && slots.build(sx, sy, x_rank)) {
            const Assignment assignment{agents.coordinates.get(), agents.coordinates.get() + n,
                                        slots.coordinates.get(), slots.coordinates.get() + n,
                                        slot_of.data()};
            assignment.split(agents.orders(n), slots.orders(n), n);
            return;
        }
    }
    slot_of.assign(n, 0);
    assign_original(ax, ay, sx, sy, slot_of);
}
}  // namespace lumen::formation
