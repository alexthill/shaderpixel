//use crate::fs;
use crate::math::{Deg, Matrix4, Vector3};

use ash::vk;
use glslang::{
    self,
    Compiler, CompilerOptions, ShaderInput, ShaderStage,
};
use std::{
    borrow::Cow,
    mem::offset_of,
    path::Path,
};

#[derive(Debug, Clone)]
pub struct Shader {
    #[allow(unused)]
    path: Option<Cow<'static, Path>>,
    data: Option<Cow<'static, [u8]>>,
}

impl Shader {
    pub fn new<P: Into<Cow<'static, Path>>>(path: P) -> Self {
        Self {
            path: Some(path.into()),
            data: None,
        }
    }

    pub fn new_static(data: &'static [u8]) -> Self {
        Self {
            path: None,
            data: Some(Cow::Borrowed(data)),
        }
    }

    pub fn data(&self) -> Option<&[u8]> {
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
            let data: Vec<u32> = shader.compile()?;
            // SAFETY
            // We are deconstructing a Vec<u32> and reconstructing it as Vec<u8>
            // with len and cap times 4.
            let data: Vec<u8> = unsafe {
                let cap = data.capacity();
                let data = data.leak();
                let ptr = std::mem::transmute::<*mut u32, *mut u8>(data.as_mut_ptr());
                Vec::from_raw_parts(ptr, data.len() * 4, cap * 4)
            };
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
