#version 450
#extension GL_ARB_separate_shader_objects : enable

#define EPS 0.00001

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 cameraPos;
layout(location = 2) in float cameraDistToContainer;
layout(location = 3) in vec2 iResolution;
layout(location = 4) in float iTime;

layout(location = 0) out vec4 outColor;

const vec4 CONTAINER_COLOR = vec4(0.0, 0.0, 0.0, 0.4);
const float SUN_RADIUS = 0.2;
const float EARTH_RADIUS = 0.1;
const float MOON_RADIUS = 0.04;

void mix_colors(float d0, vec4 c0, inout float d, inout vec4 c) {
    if (d0 < d) {
        float a = (1.0 - c0.a) * c.a + c0.a;
        d = d0;
        c = vec4(((1.0 - c0.a) * c.a * c.rgb + c0.a * c0.rgb) / a, a);
    }
}

// calculates the distance from a point o in dir u to a sphere of center c and radius r
vec2 sphere(vec3 c, float r, vec3 o, vec3 u) {
    float b = 2.0 * dot(u, o - c);
    float delta = b * b - 4.0 * (dot(o - c, o - c) - r * r);
    if (delta < 0.0) {
        return vec2(-1.0, -1.0);
    }

    float d1 = (-b + sqrt(delta)) / 2.0;
    float d2 = (-b - sqrt(delta)) / 2.0;
    return d1 < d2 ? vec2(d1, d2) : vec2(d2, d1);
}

void sun(vec3 camera, vec3 dir, inout float d, inout vec4 color) {
    vec2 dists = sphere(vec3(0.0), SUN_RADIUS, camera, dir);
    //float is_in = step(0.0, dists.x);
    //is_in = abs(dFdx(is_in)) + abs(dFdy(is_in));
    //color = vec4(is_in);
    //return;
    if (dists.x >= 0.0) {
        vec3 p = camera + dists.x * dir;
        float angle = dot(dir, normalize(vec3(0.0) - p));
        mix_colors(dists.x, vec4(1.0, 1.0, angle * 0.75, 1.0), d, color);
    } else {
        dists = sphere(vec3(0.0), 0.25, camera, dir);
        if (dists.x >= 0.0) {
            float factor = clamp(0.0, 1.0, (dists.y - dists.x) * 3.0);
            mix_colors(dists.x, vec4(1.0, 0.5, 0.0, factor), d, color);
        }
    }
}

vec3 get_earth_pos() {
    return vec3(cos(iTime * 0.1), 0.0, sin(iTime * 0.1)) * 0.7;
}

void earth(vec3 camera, vec3 dir, inout float d, inout vec4 color) {
    vec3 pos = get_earth_pos();
    vec2 dists = sphere(pos, EARTH_RADIUS, camera, dir);
    if (dists.x >= 0.0) {
        vec3 p = camera + dists.x * dir;
        float angle = dot(dir, normalize(pos - p));
        float brightness = smoothstep(-0.25, 0.2, dot(normalize(p - pos), normalize(-pos)));
        vec3 col = brightness * vec3(vec2(angle * 0.25), 1.0);
        mix_colors(dists.x, vec4(col, 1.0), d, color);
    }
}

void moon(vec3 camera, vec3 dir, inout float d, inout vec4 color) {
    vec3 earth_pos = get_earth_pos();
    vec3 pos = earth_pos + vec3(cos(iTime * 0.5), 0.0, sin(iTime * 0.5)) * 0.2;
    vec2 dists = sphere(pos, MOON_RADIUS, camera, dir);
    if (dists.x >= 0.0) {
        vec3 p = camera + dists.x * dir;
        float angle = dot(dir, normalize(pos - p));
        float brightness = smoothstep(-0.25, 0.2, dot(normalize(p - pos), normalize(-pos)));
        if (sphere(earth_pos, EARTH_RADIUS, p, -normalize(p)).x > 0.0) {
            brightness = 0.0;
        }
        vec3 col = brightness * vec3(0.4 + angle * 0.6);
        mix_colors(dists.x, vec4(col, 1.0), d, color);
    }
}

void main() {
    vec3 dir = normalize(fragPos - cameraPos);
    float d = 10000.0;
    vec4 color = CONTAINER_COLOR;
    moon(cameraPos, dir, d, color);
    earth(cameraPos, dir, d, color);
    // check sun last so that transparent corona is applied correctly to objects behind
    sun(cameraPos, dir, d, color);
    outColor = color;
}
