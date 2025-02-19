use ash::vk;
use std::mem::offset_of;

pub trait Vertex {
    fn new(pos: [f32; 3], color: [f32; 3], coords: [f32; 2]) -> Self;
    fn get_binding_description() -> vk::VertexInputBindingDescription;
    fn get_attribute_descriptions() -> Vec::<vk::VertexInputAttributeDescription>;
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct VertexSimple {
    pos: [f32; 3],
}

impl Vertex for VertexSimple {
    fn new(pos: [f32; 3], _: [f32; 3], _: [f32; 2]) -> Self {
        Self { pos }
    }

    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Self>() as _)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        let position_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Self, pos) as _);
        vec![position_desc]
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct VertexColorCoords {
    pos: [f32; 3],
    color: [f32; 3],
    coords: [f32; 2],
}

impl Vertex for VertexColorCoords {
    fn new(pos: [f32; 3], color: [f32; 3], coords: [f32; 2]) -> Self {
        Self { pos, color, coords }
    }

    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Self>() as _)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        let position_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Self, pos) as _);
        let color_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Self, color) as _);
        let coords_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(Self, coords) as _);
        vec![position_desc, color_desc, coords_desc]
    }
}
