#version 450
#extension GL_ARB_separate_shader_objects : enable

#define PI 3.1415926535
#define NUM_STEPS 128
#define EPSILON 0.0001

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 cameraPos;
layout(location = 2) in float cameraDistToContainer;
layout(location = 3) in vec2 iResolution;
layout(location = 4) in float iTime;

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// based on Morgan McGuire @morgan3d <https://www.shadertoy.com/view/4dS3Wd>
float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

#define OCTAVES 10
float fbm(vec2 st) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < OCTAVES; i++) {
        value += amplitude * noise(st);
        st *= 2.0;
        amplitude *= 0.49;
    }
    return value;
}

vec2 op_union(vec2 a, vec2 b) {
    return a.x < b.x ? a : b;
}

float sdf_box(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdf_plane(vec3 p, vec3 n, float h) {
  return dot(p, n) + h;
}

vec2 sdf_scene(vec3 p) {
    float d_plane = sdf_plane(p, vec3(0.0, 1.0, 0.0), fbm(p.xz));
    float d_cube = sdf_box(p, vec3(1.0));
    vec2 scene = vec2(max(d_cube, d_plane), 0.0);

    float d_cross_v = sdf_box(p, vec3(0.001, 0.02, 0.001));
    float d_cross_h = sdf_box(p - vec3(0.0, 0.015, 0.0), vec3(0.005, 0.001, 0.001));
    float d_cross = min(d_cross_v, d_cross_h);
    scene = op_union(scene, vec2(d_cross, 1.0));

    return scene;
}

vec3 estimate_normal(vec3 p) {
    return normalize(vec3(
        sdf_scene(vec3(p.x + EPSILON, p.y, p.z)).x - sdf_scene(vec3(p.x - EPSILON, p.y, p.z)).x,
        sdf_scene(vec3(p.x, p.y + EPSILON, p.z)).x - sdf_scene(vec3(p.x, p.y - EPSILON, p.z)).x,
        sdf_scene(vec3(p.x, p.y, p.z  + EPSILON)).x - sdf_scene(vec3(p.x, p.y, p.z - EPSILON)).x
    ));
}

vec2 raymarch(vec3 pos, vec3 dir, float max_depth) {
    float depth = 0.0;
    vec2 scene;
    for (int i = 0; i < NUM_STEPS; i++) {
        scene = sdf_scene(pos + depth * dir);
        float dist = scene.x;
        if (dist < EPSILON) {
            return vec2(depth, scene.y);
        }
        depth += dist * 0.5;
        if (depth >= max_depth) {
            return vec2(max_depth, -1.0);
        }
    }
    return vec2(depth, scene.y);
}


void main() {
    vec3 dir = normalize(fragPos - cameraPos);
    vec3 pos = cameraPos + cameraDistToContainer * dir;

    float max_depth = distance(pos, fragPos);
    vec2 scene = raymarch(pos, dir, max_depth);

    if (scene.x < max_depth) {
        vec3 normal = estimate_normal(pos + scene.x * dir);

        vec3 ambient_color = vec3(0.21, 0.2, 0.2);
        vec3 diffuse_color = vec3(0.21, 0.2, 0.2);
        if (scene.y == 0.0) {
            if (abs(dot(normal, vec3(0.0, 1.0, 0.0))) < 0.85) {
                ambient_color = vec3(0.21, 0.2, 0.2);
                diffuse_color = vec3(0.21, 0.2, 0.2);
            } else {
                ambient_color = vec3(0.8);
                diffuse_color = vec3(0.4);
            }
        } else {
            ambient_color = vec3(0.4, 0.2, 0.1);
            ambient_color = vec3(0.4, 0.2, 0.1);
        }

        vec3 light_dir = normalize(vec3(1.0, 1.0, 0.0));
        float lambertian = max(dot(light_dir, normal), 0.0);
        vec3 color = ambient_color + lambertian * diffuse_color;
        outColor = vec4(color, 1.0);
    } else {
        outColor = vec4(0.0);
    }
}
