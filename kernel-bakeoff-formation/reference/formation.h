// lumen::formation: the swarm pours into a shape. Engine-free C++20, deterministic.
//
// shape_slots() lays slots on a square grid inside a silhouette (dragon, sword and hand outlines, original), with
// the spacing chosen so that exactly n slots fall inside; assign() matches agents to slots by hierarchical rank splitting:
// both point sets are split at the same rank along the longer axis of their joint bounding box, recursively, so every half
// keeps equal counts and nearby agents take nearby slots (no crossing at the top levels). It is not an optimal transport,
// but it is a bijection in O(n log n), and a pure function of the inputs: ties break by index, no floating sums.
// A swarm then seeks the slots, its own separation keeping the walk collision-free locally.
#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace lumen::formation {

// Shape names: "dragon", "sword", "hand", "disk". The silhouette spans [-960, 960] units on both axes; it is scaled so the
// slot grid has `spacing` cm, centred on (cx, cy), and turned by (cos_yaw, sin_yaw) (the shape's +y axis points along it).
// Returns false for an unknown shape, n == 0, or a count the silhouette cannot hold at a grid step of a quarter unit.
bool shape_slots(std::string_view shape, std::size_t n, float spacing, float cx, float cy, float cos_yaw, float sin_yaw,
                 std::vector<float>& sx, std::vector<float>& sy);

// slot_of[i] = the slot agent i takes; a permutation of [0, n). Throws unless the four spans share one length.
void assign(std::span<const float> ax, std::span<const float> ay, std::span<const float> sx, std::span<const float> sy,
            std::vector<std::uint32_t>& slot_of);

}  // namespace lumen::formation
