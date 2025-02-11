#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    float texture_weight;
    float time;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
} pcs;

layout(location = 0) in vec3 vPosition;

layout(location = 0) out vec3 fragPos;
layout(location = 1) flat out float iTime;

void main() {
    fragPos = vPosition;
    iTime = ubo.time;
    gl_Position = ubo.proj * ubo.view * pcs.model * vec4(vPosition, 1.0);
}
