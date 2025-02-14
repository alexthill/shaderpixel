use crate::math::Matrix4;

use ash::{vk, Device};
use glslang::{
    self,
    Compiler, CompilerOptions, ShaderInput, ShaderStage,
};
use std::{
    borrow::Cow,
    io::Cursor,
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
    path: Option<Cow<'static, Path>>,
    code: Option<Box<[u32]>>,
    module: Option<vk::ShaderModule>,
}

impl Shader {
    pub fn new<P: Into<Cow<'static, Path>>>(path: P) -> Self {
        Self {
            path: Some(path.into()),
            code: None,
            module: None,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, anyhow::Error> {
        let mut cursor = Cursor::new(bytes);
        let code = ash::util::read_spv(&mut cursor)?;
        Ok(Self {
            path: None,
            code: Some(code.into()),
            module: None,
        })
    }

    pub fn module(&self) -> Option<vk::ShaderModule> {
        self.module
    }

    pub fn ensure(&mut self, device: &Device, stage: ShaderStage) -> Result<(), anyhow::Error> {
        if self.module.is_none() {
            if self.code.is_none() {
                self.load_code(stage)?;
            }
            self.load_module(device)?;
        }
        Ok(())
    }

    pub fn reload(&mut self, device: &Device, stage: ShaderStage) -> Result<(), anyhow::Error> {
        self.load_code(stage)?;
        self.load_module(device)?;
        Ok(())
    }

    pub fn cleanup(&mut self, device: &Device) {
        if let Some(module) = self.module.take() {
            log::debug!("cleaning Shader");
            unsafe {
                device.destroy_shader_module(module, None);
            };
        }
    }

    fn load_module(&mut self, device: &Device) -> Result<(), anyhow::Error> {
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(self.code.as_ref().unwrap());
        unsafe {
            let module = Some(device.create_shader_module(&create_info, None)?);
            self.cleanup(device);
            self.module = module;
        }
        //self.code = None;
        Ok(())
    }

    fn load_code(&mut self, stage: ShaderStage) -> Result<(), anyhow::Error> {
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
        let code = shader.compile()?;
        self.code = Some(code.into());
        Ok(())
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        if self.module.is_some() {
            panic!("Shader was not cleaned up before beeing dropped");
        }
    }
}
