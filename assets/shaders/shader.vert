#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    float texture_weight;
} ubo;

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec2 vCoords;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragCoords;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(vPosition, 1.0);
    fragColor = vColor;
    fragCoords = vCoords;
}
