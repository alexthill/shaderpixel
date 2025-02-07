use crate::math::{Deg, Matrix4, Vector3};

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
        self.data = None;
        self.ensure(stage)
    }

    pub fn ensure(&mut self, stage: ShaderStage) -> Result<(), anyhow::Error> {
        if self.data.is_none() {
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
        }
        Ok(())
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
#[allow(dead_code)]
#[repr(C)]
pub struct UniformBufferObject {
    pub model: Matrix4,
    pub view: Matrix4,
    pub proj: Matrix4,
    pub texture_weight: f32,
}

impl UniformBufferObject {
    pub fn get_descriptor_set_layout_binding<'a>() -> vk::DescriptorSetLayoutBinding<'a> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
    }

    pub fn view_matrix() -> Matrix4 {
        Matrix4::look_at_rh(
            Vector3::from([0., 0., 3.]),
            Vector3::from([0., 0., 0.]),
            Vector3::from([0., 1., 0.]),
        )
    }

    pub fn model_matrix(extent_min: Vector3, extent_max: Vector3) -> Matrix4 {
        let model_sizes = extent_max - extent_min;
        let max_size = model_sizes.x().max(model_sizes.y()).max(model_sizes.z());
        let scale = Matrix4::from_scale(1. / max_size);
        let translate = Matrix4::from_translation(-extent_min - model_sizes / 2.);
        Matrix4::from_angle_y(Deg(-90.)) * scale * translate
    }
}
