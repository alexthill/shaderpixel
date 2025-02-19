#version 450
#extension GL_ARB_separate_shader_objects : enable

#define EPS 0.00001

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 cameraPos;
layout(location = 2) in float cameraDistToContainer;
layout(location = 3) in vec2 iResolution;
layout(location = 4) in float iTime;

layout(location = 0) out vec4 outColor;

const int MENGER_DEPTH = 8;
const vec4 CONTAINER_COLOR = vec4(0.0, 0.0, 0.0, 0.4);

// Calculates the intersections of the axis-aligned box defined by the corners `c1` and `c2`
// and the ray from `pos` in direction `dir`. It must be `c1` <= `c2`.
// Returns if there is indeed an intersection.
// The intersections will be stored as columns in `inters` sorted by distance to `pos`.
// The x component contains the distance to `pos` and y the id (0-5) of the face.
bool intersect_box(vec3 c1, vec3 c2, vec3 dir, vec3 pos, out mat2 inters) {
    inters = mat2(0.0);
    bool intersects = false;
    vec3 box_middle = (c1 + c2) * 0.5;
    vec3 ts = (c1 - pos) / dir;
    for (int i = 0; i < 3; ++i) {
        vec3 p = pos + ts[i] * dir;
        p[i] = box_middle[i]; // don't check the axis perpendicular to the face currently checked
        bool inside = ts[i] >= 0.0 && all(lessThanEqual(c1, p)) && all(lessThanEqual(p, c2));
        if (inside) {
            inters[int(intersects)] = vec2(ts[i], float(i));
            intersects = true;
        }
    }
    ts = (c2 - pos) / dir;
    for (int i = 0; i < 3; ++i) {
        vec3 p = pos + ts[i] * dir;
        p[i] = box_middle[i]; // don't check the axis perpendicular to the face currently checked
        bool inside = ts[i] >= 0.0 && all(lessThanEqual(c1, p)) && all(lessThanEqual(p, c2));
        if (inside) {
            inters[int(intersects)] = vec2(ts[i], float(i + 3));
            intersects = true;
        }
    }
    if (!intersects) {
        return false;
    }
    if (inters[1].x < inters[0].x) {
        inters = mat2(inters[1], inters[0]);
    }
    return true;
}

bool shrink(vec3 pos, inout float size, inout vec3 corner) {
    // assuming the size of the sponge is strictly between 1 and 3
    const float min_size = pow(3.0, -float(MENGER_DEPTH - 1));
    while (size > min_size) {
        float third = size / 3.0;
        if (third * iResolution.y / 2.0 < distance(pos, cameraPos)) {
            return true;
        }

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

vec4 menger(vec3 corner_start, float size_start, vec3 dir, vec3 pos, float plane) {
    for (int i = 0; i < MENGER_DEPTH * 4; ++i) {
        float size = size_start;
        vec3 corner = corner_start;
        if (shrink(pos, size, corner)) {
            switch (int(plane)) {
                case 0: return vec4(0.937, 0.475, 0.541, 1.0);
                case 1: return vec4(0.969, 0.663, 0.659, 1.0);
                case 2: return vec4(0.729, 0.588, 0.690, 1.0);
                case 3: return vec4(0.490, 0.510, 0.722, 1.0);
                case 4: return vec4(0.380, 0.247, 0.459, 1.0);
                case 5: return vec4(0.898, 0.765, 0.820, 1.0);
                default: return vec4(0.0);
            }
        }

        mat2 inters;
        bool intersects = intersect_box(corner, corner + size, dir, pos, inters);
        if (!intersects) {
            // this should not happen
            return vec4(1.0);
        }

        // prevent floating point precision artefacts by moving EPS into cube
        pos += (inters[1].x + EPS) * dir;
        plane = inters[1].y;

        if (any(lessThan(pos, corner_start)) || any(lessThan(corner_start + size_start, pos))) {
            return CONTAINER_COLOR;
        }
    }
    return CONTAINER_COLOR;
}

void main() {
    vec3 dir = normalize(fragPos - cameraPos);
    mat2 inters;
    bool intersects = intersect_box(vec3(-0.75), vec3(0.75), dir, cameraPos, inters);

    if (!intersects) {
        outColor = CONTAINER_COLOR;
    } else {
        vec2 inter = inters[0].x < 0.0 ? inters[1] : inters[0];
        float plane = mod(inter.y + 3.0, 6.0);
        vec3 pos = cameraPos + inter.x * dir;
        outColor = menger(vec3(-0.75), 1.5, dir, pos, plane);
    }
}
