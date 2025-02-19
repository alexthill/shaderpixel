use crate::math::{Matrix4, Vector2};

use ash::vk;

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
