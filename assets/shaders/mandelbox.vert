#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    float texture_weight;
} ubo;

layout(location = 0) in vec3 vPosition;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 cameraPos;
layout(location = 2) out float cameraDistToContainer;

void main() {
    fragPos = vPosition;
    cameraPos = -transpose(mat3(ubo.view)) * ubo.view[3].xyz;
    // assuming container the unit cube
    cameraDistToContainer = length(max(vec3(0.0), abs(cameraPos) - 1));

    gl_Position = ubo.proj * ubo.view * vec4(vPosition, 1.0);
}
