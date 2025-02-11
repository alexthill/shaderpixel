#version 450
#extension GL_ARB_separate_shader_objects : enable

#define EPS 0.00001

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 cameraPos;
layout(location = 2) in float cameraDistToContainer;

layout(location = 0) out vec4 outColor;

// Calculates the intersections of the cube defined by the corners `c1` and `c2`
// and the ray from `pos` in direction `dir`. It must be `c1` <= `c2`.
// Returns if there is indeed an intersection.
// Sets `far` to the far intersection point and `pos` to the near one if
// `pos` is not inside the cube and there are two intersections.
bool intersect_cube(vec3 c1, vec3 c2, vec3 dir, inout vec3 pos, out vec3 far) {
    vec2 intersects;
    int intersect_count = 0;
    vec3 ts = (c1 - pos) / dir;
    for (int i = 0; i < 3; ++i) {
        vec3 p = pos + ts[i] * dir;
        bool inside = ts[i] >= 0.0 && all(lessThan(c1 - EPS, p)) && all(lessThan(p, c2 + EPS));
        if (inside) {
            intersects[intersect_count] = ts[i];
            intersect_count += 1;
        }
    }
    ts = (c2 - pos) / dir;
    for (int i = 0; i < 3; ++i) {
        vec3 p = pos + ts[i] * dir;
        bool inside = ts[i] >= 0.0 && all(lessThan(c1 - EPS, p)) && all(lessThan(p, c2 + EPS));
        if (inside) {
            intersects[intersect_count] = ts[i];
            intersect_count += 1;
        }
    }
    if (intersect_count == 0) {
        return false;
    }
    if (intersect_count == 1) {
        far = pos + intersects[0] * dir;
        return true;
    }
    if (intersects[0] < intersects[1]) {
        far = pos + intersects[1] * dir;
        pos += intersects[0] * dir;
    } else {
        far = pos + intersects[0] * dir;
        pos += intersects[1] * dir;
    }
    return true;
}

bool ray_march(vec3 corner, float size, vec3 dir, inout vec3 pos) {
    float third = size / 3.0;
    vec3 corner2 = corner + size;
    vec3 far;
    bool intersects = intersect_cube(corner, corner2, dir, pos, far);
    if (!intersects) {
        return false;
    }
    bvec3 in_third = notEqual(vec3(0.0), step(corner + third, pos) * step(pos, corner2 - third));
    return int(in_third[0]) + int(in_third[1]) + int(in_third[2]) < 2;
}

void main() {
    vec3 dir = normalize(fragPos - cameraPos);
    vec3 pos = cameraPos;
    bool intersects = ray_march(vec3(-0.75), 1.5, dir, pos);

    if (intersects) {
        outColor = vec4(normalize(pos) * 0.5 + 0.5, 1.0);
    } else {
        outColor = vec4(0.0, 0.0, 0.0, 0.4);
    }
}
