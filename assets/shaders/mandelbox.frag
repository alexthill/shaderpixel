#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    float texture_weight;
} ubo;

layout(location = 0) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

const int MAX_ITERS = 30;
const int MAX_STEPS = 128;
const float MAX_DIST = 10.0;

// TODO make these configurable
const float scaleFactor = 3.0;
const int maxIterations = 10;

vec3 get_color(vec3 pos) {
    vec3 c = pos;
    vec3 v = pos;

    for (int i = 0; i < MAX_ITERS; i++) {
        if (i == maxIterations) {
            break;
        }

        // Box fold
        v = clamp(v, -1.0, 1.0) * 2.0 - v;

        // Sphere fold
        float mag = dot(v, v);
        if (mag < 0.25) {
            v = v * 4.0;
        } else if (mag < 1.0) {
            v = v / mag;
        }

        v = v * scaleFactor + c;
    }

    float fract_iter_count = log(dot(v, v));
    vec3 amplitude = vec3(0.52, 0.53, 0.91);
    vec3 phase = vec3(0.7, 1.8, 1.1);
    return 0.75 + 0.25 * sin(vec3(fract_iter_count) * amplitude + phase);
}

float dist_estimate(vec3 ray_pos, float constant1, float constant2) {
    vec3 c = ray_pos;
    vec3 v = ray_pos;
    float dr = 1.0;

    for (int i = 0; i < MAX_ITERS; i++) {
        if (i == maxIterations) {
            break;
        }

        // Box fold
        v = clamp(v, -1.0, 1.0) * 2.0 - v;

        // Sphere fold
        float mag = dot(v, v);
        if (mag < 0.25) {
            v = v * 4.0;
            dr = dr * 4.0;
        } else if (mag < 1.0) {
            v = v / mag;
            dr = dr / mag;
        }

        v = v * scaleFactor + c;
        dr = dr * abs(scaleFactor) + 1.0;
    }

    return (length(v) - constant1) / dr - constant2;
}

int ray_march(vec3 pos, vec3 ray_dir, inout float dist) {
    float constant1 = abs(scaleFactor - 1.0);
    float constant2 = pow(float(abs(scaleFactor)), float(1 - maxIterations));

    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 ray_pos = pos + ray_dir * dist;
        float de = dist_estimate(ray_pos, constant1, constant2);

        dist += de * 0.95;

        float epsilon = 0.0002;
        if (de < epsilon || dist > MAX_DIST) {
            return i + 1;
        }
    }

    return MAX_STEPS;
}

void main() {
    vec3 camera_pos = -transpose(mat3(ubo.view)) * ubo.view[3].xyz;
    float dist_to_cube = length(max(vec3(0.0), abs(camera_pos) - 1));

    vec3 ray_dir = normalize(fragPos - camera_pos);
    vec3 ray_pos = (camera_pos + ray_dir * dist_to_cube) * 4.5;

    float dist = 0;
    int steps = ray_march(ray_pos, ray_dir, dist);

    if (dist >= MAX_DIST || steps == MAX_STEPS) {
        outColor = vec4(0.0, 0.0, 0.0, 0.4);
    } else {
        vec3 hit_pos = ray_pos + dist * ray_dir;
        outColor = vec4(get_color(hit_pos), 1.0);
    }
}
