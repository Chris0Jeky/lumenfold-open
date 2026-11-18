// lumen::formation (see formation.h).
#include "formation.h"

#include <algorithm>
#include <array>
#include <bit>
#include <memory>
#include <limits>
#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#endif
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace lumen::formation {

namespace {

int shape_id(std::string_view s) {
    return s == "disk" ? 0 : s == "sword" ? 1 : s == "hand" ? 2 : s == "dragon" ? 3 : -1;
}

struct Interval {
    int begin, end; // Half-open columns in Row, closed world-coordinate bounds in BandCover.
};

struct Grid {
    double u;
    double inverse;
    int k;

    explicit Grid(double step) : u(step), inverse(1.0 / u), k(int(std::floor(960.0 / u))) {}

    // Invert the boundary for a starting guess only. Walk to the first true
    // column with the forward, rounded predicate, including its equality rule.
    template <class Predicate> int boundary(double guess, Predicate test) const {
        int i = std::clamp(int(guess * inverse), -k, k + 1);
        while (i > -k && test((i - 1) * u))
            --i;
        while (i <= k && !test(i * u))
            ++i;
        return i;
    }
};

struct Row {
    std::array<Interval, 11> intervals;
    int size;
    Row() : size(0) {}

    void add(int begin, int end) {
        if (begin >= end)
            return;
        int at = size++;
        while (at && intervals[at - 1].begin > begin) {
            intervals[at] = intervals[at - 1];
            --at;
        }
        intervals[at] = {begin, end};
    }
};

// Floor division for a signed numerator and a positive compile-time divisor.
constexpr int floor_div(int x, int d) {
    return x / d - (x % d < 0);
}

template <int AX, int AY, int BX, int BY, int Sign> struct HalfPlane {
    static constexpr int a = Sign * (BX - AX), b = Sign * (AY - BY);
    static constexpr int c = a * AY + b * AX;
    // Inside in real arithmetic iff a*j+b*i >= c/u. The left side is an
    // integer. Even after division by the smallest u, the error in the
    // original double determinant is < 2^-20 grid units (see README).
    static constexpr double slack = 0x1p-20;
    int threshold;
    bool ambiguous;

    explicit HalfPlane(double u) {
        const double cut = c / u;
        threshold = int(std::ceil(cut - slack));
        ambiguous = b != 0 && threshold - cut <= slack;
    }

    void clip(int j, double y, Interval &r) const {
        if constexpr (b > 0)
            r.begin = std::max(r.begin, -floor_div(a * j - threshold, b));
        else if constexpr (b < 0)
            r.end = std::min(r.end, floor_div(a * j - threshold, -b) + 1);
        else if (Sign * ((BX - AX) * (y - AY)) < 0)
            r.end = r.begin;
    }

    static bool inside(double x, double y) {
        const double d = (BX - AX) * (y - AY) - (BY - AY) * (x - AX);
        return Sign > 0 ? d >= 0 : d <= 0;
    }
};

struct Raster {
    Grid g;
    Row *rows; // biased so signed row numbers index directly

    void rect(int x0, int y0, int x1, int y1) {
        const int left = g.boundary(x0, [=](double x) { return x >= x0; });
        const int right = g.boundary(x1, [=](double x) { return x > x1; });
        const int first = g.boundary(y0, [=](double y) { return y >= y0; });
        const int end = g.boundary(y1, [=](double y) { return y > y1; });
        for (int j = first; j < end; ++j)
            rows[j].add(left, right);
    }

    void ellipse(double cx, double cy, double rx, double ry) {
        // Pad the y extent by one grid row, then use the original rounded
        // squared distance on every candidate row and at the x boundaries.
        const int first = std::max(-g.k, int(std::floor((cy - ry) * g.inverse)) - 1);
        const int end = std::min(g.k + 1, int(std::ceil((cy + ry) * g.inverse)) + 2);
        for (int j = first; j < end; ++j) {
            const double dy = (j * g.u - cy) / ry, yy = dy * dy;
            if (yy > 1.0)
                continue;
            const double radius = rx * std::sqrt(1.0 - yy);
            auto inside = [=](double x) {
                const double dx = (x - cx) / rx;
                return dx * dx + yy <= 1.0;
            };
            rows[j].add(g.boundary(cx - radius, [&](double x) { return x >= cx || inside(x); }),
                        g.boundary(cx + radius, [&](double x) { return x > cx && !inside(x); }));
        }
    }

    template <int AX, int AY, int BX, int BY, int CX, int CY> void triangle() {
        constexpr int sign = ((BX - AX) * (CY - AY) > (BY - AY) * (CX - AX)) ? 1 : -1;
        const HalfPlane<AX, AY, BX, BY, sign> e0(g.u);
        const HalfPlane<BX, BY, CX, CY, sign> e1(g.u);
        const HalfPlane<CX, CY, AX, AY, sign> e2(g.u);
        const bool ambiguous = e0.ambiguous || e1.ambiguous || e2.ambiguous;
        const int first = std::max(-g.k, int(std::floor(std::min({AY, BY, CY}) * g.inverse)) - 1);
        const int end = std::min(g.k + 1, int(std::ceil(std::max({AY, BY, CY}) * g.inverse)) + 2);
        for (int j = first; j < end; ++j) {
            const double y = j * g.u;
            Interval r{-g.k, g.k + 1};
            e0.clip(j, y, r);
            e1.clip(j, y, r);
            e2.clip(j, y, r);
            if (ambiguous && r.begin < r.end) {
                // Only an endpoint can lie on an ambiguous nonhorizontal edge.
                auto inside = [&](double x) { return e0.inside(x, y) && e1.inside(x, y) && e2.inside(x, y); };
                if (!inside(r.begin * g.u))
                    ++r.begin;
                if (!inside((r.end - 1) * g.u))
                    --r.end;
            }
            rows[j].add(r.begin, r.end);
        }
    }
};

template <class Painter> constexpr void silhouette(int id, Painter &painter) {
    switch (id) {
    case 0:
        painter.ellipse(0, 0, 820, 820);
        break;
    case 1:
        painter.rect(-100, -330, 100, 700);
        painter.template triangle<-100, 700, 100, 700, 0, 960>();
        painter.rect(-440, -420, 440, -330);
        painter.rect(-60, -800, 60, -420);
        painter.ellipse(0, -835, 115, 85);
        break;
    case 2:
        painter.rect(-360, -450, 330, 230);
        painter.ellipse(0, -450, 360, 240);
        painter.ellipse(-280, 380, 90, 380);
        painter.ellipse(-85, 480, 90, 440);
        painter.ellipse(110, 460, 90, 420);
        painter.ellipse(300, 340, 85, 330);
        painter.template triangle<-350, -230, -700, 170, -500, 320>();
        painter.ellipse(-570, 180, 125, 140);
        break;
    case 3:
        painter.ellipse(100, -100, 280, 460);
        painter.ellipse(300, 430, 180, 240);
        painter.rect(340, 430, 660, 530);
        painter.template triangle<270, 600, 340, 850, 420, 570>();
        painter.template triangle<-100, 100, -900, 740, -700, 70>();
        painter.template triangle<-100, 100, -700, 70, -590, -280>();
        painter.template triangle<120, 150, 850, 790, 730, 110>();
        painter.template triangle<120, 150, 730, 110, 650, -250>();
        painter.template triangle<50, -350, -820, -780, -230, -500>();
        painter.template triangle<-820, -780, -900, -510, -650, -660>();
        painter.template triangle<80, -380, -40, -700, 250, -620>();
        break;
    }
}

// A horizontal strip cover of the fixed outline can rule out undersized
// grids without rasterizing them. Each rectangle is an outward bound; an
// ambiguous candidate still goes through the exact integer row clipping.
struct BandCover {
    int y0, y1;
    std::array<Interval, 11> intervals{};
    int size = 0;

    constexpr void add(int lo, int hi) {
        if (lo > hi)
            return;
        int j = size++;
        while (j && intervals[j - 1].begin > lo) {
            intervals[j] = intervals[j - 1];
            --j;
        }
        intervals[j] = {lo, hi};
    }
    constexpr void rect(int x0, int bottom, int x1, int top) {
        if (y1 >= bottom && y0 <= top)
            add(x0, x1);
    }
    constexpr void ellipse(int cx, int cy, int rx, int ry) {
        const int dy = std::max({0, y0 - cy, cy - y1});
        if (dy > ry)
            return;
        // Integer bisection for an outward-rounded continuous chord radius.
        const auto square = [](int x) { return std::int64_t(x) * x; };
        const auto rhs = square(rx) * (square(ry) - square(dy));
        int lo = 0, hi = rx + 1;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (square(mid) * square(ry) <= rhs)
                lo = mid;
            else
                hi = mid;
        }
        add(cx - hi - 1, cx + hi + 1);
    }
    template <int AX, int AY, int BX, int BY, int CX, int CY> constexpr void triangle() {
        constexpr int sign = ((BX - AX) * (CY - AY) > (BY - AY) * (CX - AX)) ? 1 : -1;
        int lo = -961, hi = 961;
        auto edge = [&](int x0, int ya, int x1, int yb) {
            const int a = sign * (x1 - x0), b = sign * (ya - yb);
            // Relax the determinant by one unit and maximize a*y across the strip.
            const int c = a * ya + b * x0 - 1 - std::max(a * y0, a * y1);
            if (b > 0)
                lo = std::max(lo, floor_div(c, b));
            else if (b < 0)
                hi = std::min(hi, -floor_div(c, -b));
            else if (c > 0)
                hi = -962;
        };
        edge(AX, AY, BX, BY);
        edge(BX, BY, CX, CY);
        edge(CX, CY, AX, AY);
        add(lo, hi);
    }
};

struct Bound {
    int width, pieces;
};
constexpr int strip_height = 4, strips = 1920 / strip_height;

template <int Id> constexpr Bound bound_for(int band) {
    // A unit of y overlap also covers rounding of the band index.
    BandCover cover{band * strip_height - 961, (band + 1) * strip_height - 959};
    silhouette(Id, cover);
    Bound b{};
    for (int i = 0; i < cover.size;) {
        Interval r = cover.intervals[i++];
        while (i < cover.size && cover.intervals[i].begin <= r.end)
            r.end = std::max(r.end, cover.intervals[i++].end);
        b.width += r.end - r.begin;
        b.pieces++;
    }
    return b;
}

template <int Id, std::size_t I> constexpr Bound strip_bound = bound_for<Id>(int(I));
template <int Id, std::size_t... I> constexpr auto make_cover(std::index_sequence<I...>) {
    return std::array{strip_bound<Id, I>...};
}
template <int Id> constexpr auto cover = make_cover<Id>(std::make_index_sequence<strips>{});

bool too_sparse(int id, double u, std::size_t n) {
    static constexpr const Bound *bounds[] = {cover<0>.data(), cover<1>.data(), cover<2>.data(), cover<3>.data()};
    const Bound *table = bounds[id];
    const int k = int(std::floor(960. / u));
    int width = 0, pieces = 0;
    for (int j = -k; j <= k; ++j) {
        int band = std::clamp(int((j * u + 960.) / strip_height), 0, strips - 1);
        width += table[band].width;
        pieces += table[band].pieces;
    }
    // In a covered interval there are at most width/u + 1 grid columns.
    // One extra unit in the summed widths exceeds all coordinate rounding.
    return n > std::size_t(pieces) && double(n - pieces) * u > double(width) + 1.;
}

std::size_t rasterize(int id, double u, std::vector<Row> &rows) {
    const Grid g(u);
    rows.clear();
    const std::size_t height = std::size_t(2 * g.k + 1);
    if (rows.capacity() < height)
        rows.reserve(height + height / 8);
    rows.resize(height);
    Raster raster{g, rows.data() + g.k};
    silhouette(id, raster);
    std::size_t count = 0;
    for (Row &row : rows) {
        int merged = 0;
        for (int i = 0; i < row.size;) {
            Interval span = row.intervals[i++];
            while (i < row.size && row.intervals[i].begin <= span.end)
                span.end = std::max(span.end, row.intervals[i++].end);
            row.intervals[merged++] = span;
            count += span.end - span.begin;
        }
        row.size = merged;
    }
    return count;
}

template <int Shape> double coarse_fill() {
    static const double area = [] {
        std::vector<Row> rows;
        return double(rasterize(Shape, 16., rows)) * 256.;
    }();
    return area;
}

struct Selection {
    std::size_t omitted, stride, extra, next, error = 0, seen = 0, written = 0;

    Selection(std::size_t n, std::size_t m)
        : omitted(m - n), stride(omitted ? m / omitted : 0), extra(omitted ? m % omitted : 0), next(omitted ? 0 : m) {}

    // floor((t+1)n/m) == floor(t*n/m) exactly at t=floor(h*m/(m-n)).
    // Visit the contiguous ranges between those omissions without per-point division.
    template <class Write> void visit(const Row &row, Write write) {
        for (int r = 0; r < row.size; ++r) {
            int col = row.intervals[r].begin;
            const int end = row.intervals[r].end;
            const std::size_t limit = seen + std::size_t(end - col);
            while (next < limit) {
                const int keep = int(next - seen);
                if (keep)
                    write(col, keep, written);
                written += keep;
                col += keep + 1;
                seen = next + 1;
                next += stride;
                error += extra;
                if (error >= omitted) {
                    ++next;
                    error -= omitted;
                }
            }
            if (col < end) {
                write(col, end - col, written);
                written += end - col;
            }
            seen = limit;
        }
    }
};

double add_first(double first, double second) {
    // SSE2's destination wins a collision of NaN payloads in the original transform.
    return std::isnan(first) ? first : first + second;
}

} // namespace

bool shape_slots(std::string_view shape, std::size_t n, float spacing, float cx, float cy, float cos_yaw, float sin_yaw,
                 std::vector<float> &sx, std::vector<float> &sy) {
    auto fail = [&]() {
        sx.clear();
        sy.clear();
        return false;
    };
    const int id = shape_id(shape);
    if (id < 0 || n == 0 || !(std::isfinite(spacing) && spacing > 0.f))
        return fail();
    std::vector<Row> rows;
    const double fill = id == 0 ? coarse_fill<0>() : id == 1 ? coarse_fill<1>() : id == 2 ? coarse_fill<2>() : coarse_fill<3>();
    double u = std::sqrt(fill / double(n)) * 1.02;
    std::size_t m = 0;
    for (int tries = 0; tries < 400 && u >= 0.25; ++tries, u *= 0.99)
        if (!too_sparse(id, u, n) && (m = rasterize(id, u, rows)) >= n)
            break;
    if (m < n)
        return fail();

    const int k = int(rows.size() / 2);
    const double scale = double(spacing) / u;
    const bool shared = &sx == &sy;
    const bool exceptional = !std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(cos_yaw) || !std::isfinite(sin_yaw);
    if (shared) {
        sx.clear();
        sx.reserve(n);
    } else {
        sx.resize(n);
        sy.resize(n);
    }

    // Every row uses the same column products, evaluated with the original
    // two multiplications. Axis-aligned turns can also share rounded outputs.
    const bool columns_x = !shared && !exceptional && cos_yaw == 0 && (cx != 0 || !std::signbit(cx));
    const bool columns_y = !shared && !exceptional && sin_yaw == 0 && (cy != 0 || !std::signbit(cy));
    std::vector<double> column_terms(2 * rows.size());
    double *tx = column_terms.data() + k;
    double *ty = tx + rows.size();
    std::vector<float> aligned(columns_x || columns_y ? rows.size() : 0);
    for (int i = -k; i <= k; ++i) {
        const double wx = scale * (i * u);
        tx[i] = wx * sin_yaw;
        ty[i] = wx * cos_yaw;
        if (columns_x)
            aligned[i + k] = float(cx + tx[i]);
        else if (columns_y)
            aligned[i + k] = float(cy - ty[i]);
    }

    Selection selected(n, m);
    for (int j = -k; j <= k; ++j) {
        const Row &row = rows[j + k];
        if (!row.size)
            continue;
        const double wy = scale * (j * u);
        const double bx = exceptional ? add_first(wy * cos_yaw, cx) : cx + wy * cos_yaw;
        const double by = exceptional ? add_first(wy * sin_yaw, cy) : cy + wy * sin_yaw;
        // At a zero row base the sign of an added or subtracted zero can matter.
        const bool copy_x = columns_x && by != 0;
        const bool copy_y = columns_y && !columns_x && bx != 0;
        if (copy_x || copy_y) {
            float *varying = copy_x ? sx.data() : sy.data();
            float *constant = copy_x ? sy.data() : sx.data();
            const std::size_t first = selected.written;
            selected.visit(
                row, [&](int i, int length, std::size_t out) { std::copy_n(aligned.data() + (i + k), length, varying + out); });
            std::fill(constant + first, constant + selected.written, float(copy_x ? by : bx));
        } else if (!shared && !exceptional) {
            selected.visit(row, [&](int i, int length, std::size_t out) {
                for (int t = 0; t < length; ++t) {
                    sx[out + t] = float(bx + tx[i + t]);
                    sy[out + t] = float(by - ty[i + t]);
                }
            });
        } else {
            selected.visit(row, [&](int i, int length, std::size_t out) {
                for (int t = 0; t < length; ++t) {
                    const float x = float(add_first(bx, tx[i + t]));
                    const float y = float(std::isnan(by) ? by : by - ty[i + t]);
                    if (shared) {
                        sx.push_back(x);
                        sy.push_back(y);
                    } else {
                        sx[out + t] = x;
                        sy[out + t] = y;
                    }
                }
            });
        }
    }
    return sx.size() == n;
}

namespace {

// Preserve the library's original selection and traversal when the coordinates
// are unordered (NaNs), or when the public index type wraps.
void assign_original(std::span<const float> ax, std::span<const float> ay, std::span<const float> sx, std::span<const float> sy,
                     std::vector<std::uint32_t> &slot_of) {
    const std::size_t n = ax.size();
    std::vector<std::uint32_t> A(n), S(n);
    for (std::size_t i = 0; i < n; ++i) {
        A[i] = std::uint32_t(i);
        S[i] = std::uint32_t(i);
    }
    slot_of.assign(n, 0);
    std::vector<std::pair<std::size_t, std::size_t>> stack;
    if (n)
        stack.push_back({0, n});
    while (!stack.empty()) {
        const auto [lo, hi] = stack.back();
        stack.pop_back();
        if (hi - lo == 1) {
            slot_of[A[lo]] = S[lo];
            continue;
        }
        float x0 = ax[A[lo]], x1 = x0, y0 = ay[A[lo]], y1 = y0;
        for (std::size_t k = lo; k < hi; ++k) {
            x0 = std::min({x0, ax[A[k]], sx[S[k]]});
            x1 = std::max({x1, ax[A[k]], sx[S[k]]});
            y0 = std::min({y0, ay[A[k]], sy[S[k]]});
            y1 = std::max({y1, ay[A[k]], sy[S[k]]});
        }
        const bool by_x = x1 - x0 >= y1 - y0;
        const std::size_t mid = lo + (hi - lo) / 2;
        auto split = [&](std::vector<std::uint32_t> &v, std::span<const float> px, std::span<const float> py) {
            std::nth_element(v.begin() + std::ptrdiff_t(lo), v.begin() + std::ptrdiff_t(mid), v.begin() + std::ptrdiff_t(hi),
                             [&](std::uint32_t a, std::uint32_t b) {
                                 const float ka = by_x ? px[a] : py[a], kb = by_x ? px[b] : py[b];
                                 return ka != kb ? ka < kb : a < b;
                             });
        };
        split(A, ax, ay);
        split(S, sx, sy);
        stack.push_back({mid, hi});
        stack.push_back({lo, mid});
    }
}

using Index = std::uint32_t;
using Key = std::uint64_t;

Index ordered_bits(float value) {
    Index bits = std::bit_cast<Index>(value);
    if ((bits << 1) == 0)
        bits = 0; // both signs of zero tie by index
    return bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
}

Key order_key(float value, Index id) {
    return (Key(ordered_bits(value)) << 32) | id;
}

struct Entry {
    Key other;
    Index id() const { return Index(other); }
};

struct Axis {
    Entry *order;
    Entry *scratch;
    const float *values;
    float value(std::size_t i) const { return values[order[i].id()]; }
};
struct Set {
    Axis x, y;
};

// Stable radix passes start in index order, giving exactly (value, index)
// order. An already ordered axis and a constant digit need no scatter pass.
bool prepare_axis(Axis &axis, std::span<const float> values, std::span<const float> other) {
    const std::size_t n = values.size();
    if (std::is_sorted(values.begin(), values.end())) {
        bool finite = true;
        for (std::size_t i = 0; i < n; ++i) {
            finite &= std::isfinite(values[i]);
            axis.order[i] = {order_key(other[i], Index(i))};
        }
        return finite;
    }
    Index hist[4][256]{};
    bool finite = true;
    for (std::size_t i = 0; i < n; ++i) {
        const float value = values[i];
        const Index bits = ordered_bits(value);
        axis.order[i] = {(Key(bits) << 32) | Index(i)};
        finite &= std::isfinite(value);
        for (int d = 0; d < 4; ++d)
            ++hist[d][(bits >> (8 * d)) & 255];
    }
    if (!finite)
        return false;
    const Key first = axis.order[0].other;
    int last = 3;
    while (last > 0 && hist[last][(first >> (32 + 8 * last)) & 255] == n)
        --last;
    for (int d = 0; d <= last; ++d) {
        auto &h = hist[d];
        const int shift = 32 + 8 * d;
        if (h[(first >> shift) & 255] == n)
            continue;
        Index start = 0;
        for (Index &v : h) {
            const Index count = v;
            v = start;
            start += count;
        }
        if (d == last) {
            // Fuse conversion to opposite-axis keys with the final scatter.
            for (std::size_t i = 0; i < n; ++i) {
                const Key key = axis.order[i].other;
                const Index id = Index(key);
                axis.scratch[h[(key >> shift) & 255]++] = {order_key(other[id], id)};
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                const Key key = axis.order[i].other;
                axis.scratch[h[(key >> shift) & 255]++] = {key};
            }
        }
        std::swap(axis.order, axis.scratch);
    }
    return true;
}
// The median entry gives the original nth_element cut exactly. Slice the
// chosen order in place and stably partition only the opposite order.
void distribute(const Entry *a, const Entry *s, Entry *da, Entry *ds, Key ca, Key cs, std::size_t mid, std::size_t n) {
    std::size_t la = 0, ls = 0, ra = mid, rs = mid;
    for (std::size_t i = 0; i < n; ++i) {
        const Entry va = a[i], vs = s[i];
        const bool alow = va.other < ca, slow = vs.other < cs;
        da[alow ? la : ra] = va;
        ds[slow ? ls : rs] = vs;
        la += alow;
        ra += !alow;
        ls += slow;
        rs += !slow;
    }
}
struct Extent {
    int min, max;
};

// Conversion back to float must reproduce every input. Only then can integer
// offsets and widths stand in for these coordinates.
bool integer_extent(std::span<const float> values, Extent &e) {
    float lo = values[0], hi = lo;
    std::size_t i = 0;
#if defined(_M_X64) || defined(__SSE2__)
    __m128 low = _mm_set1_ps(lo), high = low;
    __m128 equal = _mm_castsi128_ps(_mm_set1_epi32(-1));
    for (; i + 4 <= values.size(); i += 4) {
        const __m128 v = _mm_loadu_ps(values.data() + i);
        equal = _mm_and_ps(equal, _mm_cmpeq_ps(v, _mm_cvtepi32_ps(_mm_cvttps_epi32(v))));
        low = _mm_min_ps(low, v);
        high = _mm_max_ps(high, v);
    }
    if (_mm_movemask_ps(equal) != 15)
        return false;
    float lanes[4];
    _mm_storeu_ps(lanes, low);
    lo = *std::min_element(lanes, lanes + 4);
    _mm_storeu_ps(lanes, high);
    hi = *std::max_element(lanes, lanes + 4);
#endif
    for (; i < values.size(); ++i) {
        const float v = values[i];
        if (!(v >= -2147483648.f && v < 2147483648.f) || float(int(v)) != v)
            return false;
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    e = {int(lo), int(hi)};
    return true;
}

// High 16 bits hold the opposite coordinate, the middle 32 the index, and
// the low 16 this coordinate. Comparing whole words splits by (opposite, id).
void pack_axis(Axis axis, std::span<const float> v, std::span<const float> w, int base, int otherbase, std::span<Index> hist) {
    const std::size_t n = v.size();
    auto word = [&](std::size_t i) { return (Key(int(w[i]) - otherbase) << 48) | (Key(i) << 16) | Index(int(v[i]) - base); };
    if (std::is_sorted(v.begin(), v.end())) {
        for (std::size_t i = 0; i < n; ++i)
            axis.order[i] = {word(i)};
        return;
    }
    std::fill(hist.begin(), hist.end(), 0u);
    for (float q : v)
        ++hist[int(q) - base];
    Index pos = 0;
    for (Index &at : hist) {
        const Index count = at;
        at = pos;
        pos += count;
    }
    // Scatter by coarse band first, keeping the exact counting scatter local.
    constexpr int lowbits = 8;
    Index coarse[1 << (16 - lowbits)];
    for (std::size_t k = 0; k < hist.size(); k += 1 << lowbits)
        coarse[k >> lowbits] = hist[k];
    for (std::size_t i = 0; i < n; ++i) {
        const Key k = word(i);
        axis.scratch[coarse[std::uint16_t(k) >> lowbits]++] = {k};
    }
    for (std::size_t i = 0; i < n; ++i) {
        const Key k = axis.scratch[i].other;
        axis.order[hist[std::uint16_t(k)]++] = {k};
    }
}

void pack_orders(Set set, std::span<const float> x, std::span<const float> y, int xmin, int ymin, std::span<Index> xhist,
                 std::span<Index> yhist) {
    pack_axis(set.x, x, y, xmin, ymin, xhist);
    pack_axis(set.y, y, x, ymin, xmin, yhist);
}
bool pack(Set a, Set s, std::span<const float> ax, std::span<const float> ay, std::span<const float> sx,
          std::span<const float> sy) {
    Extent ex, ey, fx, fy;
    if (!integer_extent(ax, ex) || !integer_extent(ay, ey) || !integer_extent(sx, fx) || !integer_extent(sy, fy))
        return false;
    const int xmin = std::min(ex.min, fx.min), ymin = std::min(ey.min, fy.min);
    const auto nx = std::int64_t(std::max(ex.max, fx.max)) - xmin + 1;
    const auto ny = std::int64_t(std::max(ey.max, fy.max)) - ymin + 1;
    if (nx > 65536 || ny > 65536 || nx + ny > 4 * std::int64_t(ax.size()))
        return false;
    auto hist = std::make_unique_for_overwrite<Index[]>(std::size_t(nx + ny));
    const std::span<Index> hx(hist.get(), std::size_t(nx)), hy(hist.get() + nx, std::size_t(ny));
    pack_orders(a, ax, ay, xmin, ymin, hx, hy);
    pack_orders(s, sx, sy, xmin, ymin, hx, hy);
    return true;
}

template <bool Packed> auto value(Axis axis, std::size_t i) {
    if constexpr (Packed)
        return int(std::uint16_t(axis.order[i].other));
    else
        return axis.value(i);
}

template <bool Packed> Key cut(Axis axis, std::size_t i) {
    const Key k = axis.order[i].other;
    if constexpr (Packed)
        return (k << 48) | (k & 0x0000ffffffff0000ull);
    else
        return order_key(axis.value(i), Index(k));
}

template <bool Packed> Index id(Entry item) {
    if constexpr (Packed)
        return Index(item.other >> 16);
    else
        return Index(item.other);
}

template <bool Packed> auto opposite_value(Entry e) {
    if constexpr (Packed)
        return int(e.other >> 48);
    else {
        Index b = Index(e.other >> 32);
        return std::bit_cast<float>(b ^ (0x80000000u | ((b >> 31) - 1)));
    }
}

// Two points in parent-axis order can be matched directly. The child either
// keeps that order or uses the opposite keys already stored with the points.
template <bool Packed, bool ParentX> void pair(Axis a, Axis s, std::size_t i, Index *out) {
    const Entry a0 = a.order[i], a1 = a.order[i + 1], s0 = s.order[i], s1 = s.order[i + 1];
    const auto along =
        std::max(value<Packed>(a, i + 1), value<Packed>(s, i + 1)) - std::min(value<Packed>(a, i), value<Packed>(s, i));
    const auto a0v = opposite_value<Packed>(a0), a1v = opposite_value<Packed>(a1);
    const auto s0v = opposite_value<Packed>(s0), s1v = opposite_value<Packed>(s1);
    const auto across = std::max(std::max(a0v, a1v), std::max(s0v, s1v)) - std::min(std::min(a0v, a1v), std::min(s0v, s1v));
    const bool change = ParentX ? across > along : across >= along;
    const unsigned flip = change & ((a0.other > a1.other) != (s0.other > s1.other));
    out[id<Packed>(a0)] = id<Packed>(s.order[i + flip]);
    out[id<Packed>(a1)] = id<Packed>(s.order[i + 1 - flip]);
}

template <bool Packed, bool ParentX> void finish(Axis a, Axis s, std::size_t n, Index *out) {
    if (n == 2) {
        out[id<Packed>(a.order[0])] = id<Packed>(s.order[0]);
        out[id<Packed>(a.order[1])] = id<Packed>(s.order[1]);
    } else {
        if (n == 3)
            out[id<Packed>(a.order[0])] = id<Packed>(s.order[0]);
        else
            pair<Packed, ParentX>(a, s, 0, out);
        pair<Packed, ParentX>(a, s, n - 2, out);
    }
}

struct Buffer {
    Entry *a;
    Entry *s;
};
struct Workspace {
    Buffer buffers[3];
    const float *ax;
    const float *ay;
    const float *sx;
    const float *sy;
    Index *out;
};

template <bool Packed> void run_match(const Workspace &w, std::size_t lo, std::size_t n, int X, int Y) {
    // Two of the three buffers hold sorted projections. The old opposite
    // projection becomes the spare after partitioning into this one.
    const int tmp = 3 - X - Y;
    const Axis ax{w.buffers[X].a + lo, nullptr, w.ax}, ay{w.buffers[Y].a + lo, nullptr, w.ay};
    const Axis sx{w.buffers[X].s + lo, nullptr, w.sx}, sy{w.buffers[Y].s + lo, nullptr, w.sy};
    const auto ax0 = value<Packed>(ax, 0), ax1 = value<Packed>(ax, n - 1);
    const auto ay0 = value<Packed>(ay, 0), ay1 = value<Packed>(ay, n - 1);
    const auto sx0 = value<Packed>(sx, 0), sx1 = value<Packed>(sx, n - 1);
    const auto sy0 = value<Packed>(sy, 0), sy1 = value<Packed>(sy, n - 1);
    const auto dx = std::max(ax1, sx1) - std::min(ax0, sx0);
    const auto dy = std::max(ay1, sy1) - std::min(ay0, sy0);
    // Every descendant contains points from both sets. If the separation on
    // one axis dominates the full other extent, all its splits use that axis.
    const bool all_x = (ax0 - sx1 >= dy) || (sx0 - ax1 >= dy) || dy == 0;
    const bool all_y = (ay0 - sy1 > dx) || (sy0 - ay1 > dx);
    if (all_x || all_y) {
        const Entry *agents = all_x ? ax.order : ay.order;
        const Entry *slots = all_x ? sx.order : sy.order;
        for (std::size_t i = 0; i < n; ++i)
            w.out[id<Packed>(agents[i])] = id<Packed>(slots[i]);
        return;
    }
    const bool by_x = dx >= dy;
    if (n <= 4) {
        if (by_x)
            finish<Packed, true>(ax, sx, n, w.out);
        else
            finish<Packed, false>(ay, sy, n, w.out);
        return;
    }
    const std::size_t mid = n / 2;
    const auto da = w.buffers[tmp].a + lo, ds = w.buffers[tmp].s + lo;
    if (by_x) {
        distribute(ay.order, sy.order, da, ds, cut<Packed>(ax, mid), cut<Packed>(sx, mid), mid, n);
        run_match<Packed>(w, lo, mid, X, tmp);
        run_match<Packed>(w, lo + mid, n - mid, X, tmp);
    } else {
        distribute(ax.order, sx.order, da, ds, cut<Packed>(ay, mid), cut<Packed>(sy, mid), mid, n);
        run_match<Packed>(w, lo, mid, tmp, Y);
        run_match<Packed>(w, lo + mid, n - mid, tmp, Y);
    }
}

} // namespace

void assign(std::span<const float> ax, std::span<const float> ay, std::span<const float> sx, std::span<const float> sy,
            std::vector<std::uint32_t> &slot_of) {
    const std::size_t n = ax.size();
    if (ay.size() != n || sx.size() != n || sy.size() != n)
        throw std::invalid_argument("lumen::formation::assign: one length for agents and slots");
    if (n < 32 || n > UINT32_MAX || n > std::numeric_limits<std::size_t>::max() / (6 * sizeof(Entry))) {
        assign_original(ax, ay, sx, sy, slot_of);
        return;
    }
    // The two point sets each need two orders and one spare.
    auto entries = std::make_unique_for_overwrite<Entry[]>(6 * n);
    Entry *p = entries.get();
    Set a{{p, p + 2 * n, ax.data()}, {p + n, p + 2 * n, ay.data()}};
    Set s{{p + 3 * n, p + 5 * n, sx.data()}, {p + 4 * n, p + 5 * n, sy.data()}};
    const bool packed = pack(a, s, ax, ay, sx, sy);
    if (!packed) {
        auto prepare = [&](Set &points, std::span<const float> x, std::span<const float> y) {
            if (!prepare_axis(points.x, x, y))
                return false;
            points.y.scratch = points.x.scratch;
            return prepare_axis(points.y, y, x);
        };
        if (!prepare(a, ax, ay) || !prepare(s, sx, sy)) {
            assign_original(ax, ay, sx, sy, slot_of);
            return;
        }
    }
    slot_of.resize(n);
    const Workspace w{{{a.x.order, s.x.order}, {a.y.order, s.y.order}, {a.y.scratch, s.y.scratch}},
                      ax.data(),
                      ay.data(),
                      sx.data(),
                      sy.data(),
                      slot_of.data()};
    if (packed)
        run_match<true>(w, 0, n, 0, 1);
    else
        run_match<false>(w, 0, n, 0, 1);
}

} // namespace lumen::formation
