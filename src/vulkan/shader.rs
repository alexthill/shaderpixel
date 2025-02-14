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
    sync::{Arc, RwLock},
    sync::mpsc::SyncSender,
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

pub struct Shader {
    inner: Arc<RwLock<ShaderInner>>,
}

impl Shader {
    pub fn set_hot_reload(&mut self, sender: SyncSender<Shader>) {
        let mut inner = self.inner.write().unwrap();
        if inner.compile_sender.is_some() {
            return;
        }
        inner.compile_sender = Some(sender.clone());
        drop(inner);
        sender.send(self.clone()).unwrap();
    }

    pub fn reload(&self, device: &Device) -> bool {
        if self.cleanup(device) {
            log::debug!("queueing shader for reloading");
            let inner = self.inner.read().unwrap();
            let sender = inner.compile_sender.clone();
            drop(inner);
            if let Some(sender) = sender {
                if sender.send(self.clone()).is_err() {
                    return false;
                }
            }
            true
        } else {
            log::debug!("shader already queued");
            false
        }
    }

    pub fn module(&self, device: &Device) -> Option<vk::ShaderModule> {
        let inner = self.inner.read().unwrap();
        let module = inner.module;
        if module.is_none() && inner.code.is_some() {
            drop(inner);
            let mut inner = self.inner.write().unwrap();
            inner.load_module(device).expect("invalid shader code");
            inner.module
        } else {
            module
        }
    }

    pub fn load_code(&self) {
        let inner = self.inner.read().unwrap();
        let stage = inner.stage;
        let path = inner.path.clone();
        drop(inner);
        match ShaderInner::load_code(stage, path.as_ref()) {
            Ok(code) => {
                let mut inner = self.inner.write().unwrap();
                inner.code = Some(code);
                inner.module = None;
            }
            Err(err) => {
                log::error!("Error loading shader code:\n{err:#}");
            }
        }
    }

    pub fn cleanup(&self, device: &Device) -> bool {
        let mut inner = self.inner.write().unwrap();
        inner.cleanup(device)
    }
}

impl Clone for Shader {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl From<ShaderInner> for Shader {
    fn from(value: ShaderInner) -> Self {
        Self {
            inner: Arc::new(RwLock::new(value)),
        }
    }
}

pub struct ShaderInner {
    stage: ShaderStage,
    path: Option<Cow<'static, Path>>,
    code: Option<Box<[u32]>>,
    module: Option<vk::ShaderModule>,
    compile_sender: Option<SyncSender<Shader>>,
}

impl ShaderInner {
    pub fn new(stage: ShaderStage) -> Self {
        Self {
            stage,
            path: None,
            code: None,
            module: None,
            compile_sender: None,
        }
    }

    pub fn path<P: Into<Cow<'static, Path>>>(mut self, path: P) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn bytes(mut self, bytes: &[u8]) -> Result<Self, anyhow::Error> {
        let mut cursor = Cursor::new(bytes);
        let code = ash::util::read_spv(&mut cursor)?;
        self.code = Some(code.into());
        Ok(self)
    }

    fn load_module(&mut self, device: &Device) -> Result<(), anyhow::Error> {
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(self.code.as_ref().unwrap());
        unsafe {
            let module = Some(device.create_shader_module(&create_info, None)?);
            self.cleanup(device);
            self.module = module;
        }
        self.code = None;
        Ok(())
    }

    fn load_code(stage: ShaderStage, path: Option<&Cow<'static, Path>>)
        -> Result<Box<[u32]>, anyhow::Error>
    {
        let input_path = path
            .expect("shader must have a path set to load it")
            .to_str()
            .unwrap();
        log::debug!("compiling shader {input_path} of stage {:?}", stage);

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
        Ok(code.into())
    }

    fn cleanup(&mut self, device: &Device) -> bool {
        if let Some(module) = self.module.take() {
            log::debug!("cleaning Shader");
            unsafe {
                device.destroy_shader_module(module, None);
            };
            true
        } else {
            false
        }
    }
}

impl Drop for ShaderInner {
    fn drop(&mut self) {
        if self.module.is_some() {
            panic!("Shader was not cleaned up before beeing dropped");
        }
    }
}
