use crate::math::Matrix4;

use ash::vk;
use glslang::{
    self,
    Compiler, CompilerOptions, ShaderInput, ShaderStage,
};
use std::{
    borrow::Cow,
    io::Cursor,
    mem::offset_of,
    path::Path,
};

pub struct Shaders {
    pub main_vert: Shader,
    pub main_frag: Shader,
    pub cube_vert: Shader,
    pub cube_frag: Shader,
    pub shaders_art: Vec<ShaderArt>,
}

pub struct ShaderArt {
    pub is_3d: bool,
    pub vert: Shader,
    pub frag: Shader,
    pub model_matrix: Matrix4,
}

#[derive(Debug, Clone)]
pub struct Shader {
    #[allow(unused)]
    path: Option<Cow<'static, Path>>,
    data: Option<Box<[u32]>>,
}

impl Shader {
    pub fn new<P: Into<Cow<'static, Path>>>(path: P) -> Self {
        Self {
            path: Some(path.into()),
            data: None,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, anyhow::Error> {
        let mut cursor = Cursor::new(bytes);
        let data = ash::util::read_spv(&mut cursor)?;
        Ok(Self {
            path: None,
            data: Some(data.into()),
        })
    }

    pub fn data(&self) -> Option<&[u32]> {
        self.data.as_deref()
    }

    pub fn reload(&mut self, stage: ShaderStage) -> Result<(), anyhow::Error> {
        let input_path = self.path
            .as_ref()
            .expect("shader must have a path set to load it")
            .to_str()
            .unwrap();
        log::debug!("compiling shader {input_path} of stage {stage:?}");

        let source = std::fs::read_to_string(input_path)?.into();
        let compiler = Compiler::acquire().unwrap();
        let input = ShaderInput::new(
            &source,
            stage,
            &CompilerOptions::default(),
            None,
            None,
        )?;
        let shader = compiler.create_shader(input)?;
        let data = shader.compile()?;
        self.data = Some(data.into());
        Ok(())
    }

    pub fn ensure(&mut self, stage: ShaderStage) -> Result<(), anyhow::Error> {
        if self.data.is_none() {
            self.reload(stage)
        } else {
            Ok(())
        }
    }
}

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
    pub texture_weight: f32,
    pub time: f32,
}

impl UniformBufferObject {
    pub fn get_descriptor_set_layout_binding<'a>() -> vk::DescriptorSetLayoutBinding<'a> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
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
