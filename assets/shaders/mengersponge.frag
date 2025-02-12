#version 450
#extension GL_ARB_separate_shader_objects : enable

#define EPS 0.00001

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 cameraPos;
layout(location = 2) in float cameraDistToContainer;

layout(location = 0) out vec4 outColor;

const int MENGER_DEPTH = 2;
const vec4 CONTAINER_COLOR = vec4(0.0, 0.0, 0.0, 0.4);

// Calculates the intersections of the cube defined by the corners `c1` and `c2`
// and the ray from `pos` in direction `dir`. It must be `c1` <= `c2`.
// Returns if there is indeed an intersection.
// The distances from `pos` to the intersections will be storted sorted in `dists`.
// One of the distances may be negative or zero if inside the cube.
bool intersect_cube(vec3 c1, vec3 c2, vec3 dir, vec3 pos, out vec2 dists) {
    dists = vec2(0.0);
    bool intersects = false;
    vec3 ts = (c1 - pos) / dir;
    for (int i = 0; i < 3; ++i) {
        vec3 p = pos + ts[i] * dir;
        bool inside = ts[i] >= 0.0 && all(lessThan(c1 - EPS, p)) && all(lessThan(p, c2 + EPS));
        if (inside) {
            dists[int(intersects)] = ts[i];
            intersects = true;
        }
    }
    ts = (c2 - pos) / dir;
    for (int i = 0; i < 3; ++i) {
        vec3 p = pos + ts[i] * dir;
        bool inside = ts[i] >= 0.0 && all(lessThan(c1 - EPS, p)) && all(lessThan(p, c2 + EPS));
        if (inside) {
            dists[int(intersects)] = ts[i];
            intersects = true;
        }
    }
    if (!intersects) {
        return false;
    }
    dists = vec2(min(dists[0], dists[1]), max(dists[0], dists[1]));
    return true;
}

bool shrink(vec3 pos, inout float size, inout vec3 corner) {
    // assuming the size of the sponge is strictly between 1 and 3
    const float min_size = pow(3.0, -float(MENGER_DEPTH - 1));
    while (size > min_size) {
        float third = size / 3.0;
        vec3 thirds = step(corner + third, pos) + step(corner + 2.0 * third, pos);
        bvec3 in_middle = equal(thirds, vec3(1.0));

        corner += third * thirds;
        size = third;

        if (int(in_middle[0]) + int(in_middle[1]) + int(in_middle[2]) >= 2) {
            return false;
        }
    }
    return true;
}

vec4 menger(vec3 corner_start, float size_start, vec3 dir, vec3 pos) {
    for (int i = 0; i < MENGER_DEPTH * 4; ++i) {
        float size = size_start;
        vec3 corner = corner_start;
        if (shrink(pos, size, corner)) {
            return vec4(normalize(pos) * 0.5 + 0.5, 1.0);
        }

        vec2 dists;
        bool intersects = intersect_cube(corner, corner + size, dir, pos, dists);
        if (!intersects) {
            // this should not happen
            return vec4(1.0);
        }

        // prevent floating point precision artefacts by moving EPS into cube
        pos += (dists[1] + EPS) * dir;

        if (any(lessThan(pos, corner_start)) || any(lessThan(corner_start + size_start, pos))) {
            return CONTAINER_COLOR;
        }
    }
    return CONTAINER_COLOR;
}

void main() {
    vec3 dir = normalize(fragPos - cameraPos);
    vec2 dists;
    bool intersects = intersect_cube(vec3(-0.75), vec3(0.75), dir, cameraPos, dists);

    if (!intersects) {
        outColor = CONTAINER_COLOR;
    } else {
        float dist_to_cube = dists[0] < 0.0 ? dists[1] : dists[0];
        vec3 pos = cameraPos + dist_to_cube * dir;
        outColor = menger(vec3(-0.75), 1.5, dir, pos);
    }
}
