use crate::math::{Matrix4, Vector2};

use ash::vk;
use std::{
    mem::offset_of,
};

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
#[repr(C)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
    pub coords: [f32; 2],
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as _)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let position_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, pos) as _);
        let color_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, color) as _);
        let coords_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, coords) as _);
        [position_desc, color_desc, coords_desc]
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct UniformBufferObject {
    pub model: Matrix4,
    pub view: Matrix4,
    pub proj: Matrix4,
    pub resolution: Vector2,
    pub texture_weight: f32,
    pub time: f32,
}

impl UniformBufferObject {
    pub fn get_descriptor_set_layout_binding<'a>() -> vk::DescriptorSetLayoutBinding<'a> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PushConstants {
    pub model: Matrix4,
}

impl PushConstants {
    pub fn get_push_constant_range() -> vk::PushConstantRange {
        vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: size_of::<Self>() as _,
        }
    }
}

impl Default for PushConstants {
    fn default() -> Self {
        Self {
            model: Matrix4::unit(),
        }
    }
}
