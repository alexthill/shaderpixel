use crate::math::Matrix4;

use ash::{vk, Device};
use glslang::{
    self,
    Compiler, CompilerOptions, ShaderInput, ShaderStage,
};
use notify_debouncer_full::{new_debouncer, notify};
use std::{
    collections::{HashMap, HashSet},
    io::Cursor,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    sync::mpsc::{self, Sender},
    time::Duration,
    thread,
};

const DEBOUNCE_TIME: Duration = Duration::from_millis(500);

pub struct Shaders {
    pub main_vert: Shader,
    pub main_frag: Shader,
    pub cube_vert: Shader,
    pub cube_frag: Shader,
    pub shaders_art: Vec<ShaderArt>,
}

impl Shaders {
    pub fn watch_art(&self) {
        let shaders_by_path = self.shaders_art.iter()
            .flat_map(|shader| [shader.vert.clone(), shader.frag.clone()])
            .filter_map(|shader| shader.path()
                        .and_then(|path| std::fs::canonicalize(&path).ok())
                        .map(|path| (path, shader)))
            .collect::<HashMap<_, _>>();

        thread::spawn(move || {
            let (tx, rx) = mpsc::channel();
            let mut debouncer = match new_debouncer(DEBOUNCE_TIME, None, tx) {
                Ok(debouncer) => debouncer,
                Err(err) => {
                    log::error!("failed to create file watcher: {err}");
                    return;
                }
            };
            let dirs_to_watch = shaders_by_path.keys()
                .filter_map(|path| path.parent())
                .collect::<HashSet<_>>();
            for path in dirs_to_watch {
                if let Err(err) = debouncer.watch(path, notify::RecursiveMode::Recursive) {
                    log::error!("failed to watch {}: {err}", path.display());
                } else {
                    log::debug!("watching file {}", path.display());
                }
            }
            for res in rx {
                match res {
                    Ok(events) => {
                        for event in events {
                            use notify::EventKind::*;
                            use notify::event::{AccessKind::*, AccessMode::*, ModifyKind::*};

                            //log::info!("event: {:?}", event);
                            if let Access(Close(Write)) | Modify(Data(_)) = event.kind {
                                for path in &event.paths {
                                    if let Some(shader) = shaders_by_path.get(path) {
                                        if let Some(path) = shader.path() {
                                            log::info!("shader changed {}", path.display());
                                            let Ok(mut inner) = shader.inner.write() else {
                                                log::error!("Lock poisoned");
                                                continue;
                                            };
                                            inner.code_has_changed = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => log::info!("watch error: {:?}", e),
                }
            }
        });
    }
}

pub struct ShaderArt {
    pub name: String,
    pub is_3d: bool,
    pub vert: Shader,
    pub frag: Shader,
    pub model_matrix: Matrix4,
}

pub struct Shader {
    inner: Arc<RwLock<ShaderInner>>,
}

impl Shader {
    pub fn path(&self) -> Option<PathBuf> {
        self.inner.read().ok()?.path.clone()
    }

    pub fn set_hot_reload(&mut self, sender: Sender<Shader>) {
        let mut inner = self.inner.write().unwrap();
        if inner.compile_sender.is_some() {
            return;
        }
        inner.is_compiling = true;
        inner.compile_sender = Some(sender.clone());
        drop(inner);
        sender.send(self.clone()).unwrap();
    }

    pub fn code_has_changed(&self) -> bool {
        self.inner.read().map(|inner| inner.code_has_changed).unwrap_or(false)
    }

    pub fn reload(&self, device: &Device, forced: bool) -> bool {
        let path = self.inner.read().unwrap()
            .path.as_ref().expect("shader must have a path set to load it").clone();
        let mut inner = self.inner.write().unwrap();
        if inner.is_compiling {
            return true;
        }
        if !inner.code_has_changed && !forced {
            return false;
        }

        // reset code_has_changed here so we dont try to recompile infinitly if an error happens
        inner.code_has_changed = false;

        let Some(sender) = inner.compile_sender.clone() else {
            log::error!("tried to queue Shader without sender {}", path.display());
            return false;
        };
        match sender.send(self.clone()) {
            Ok(_) => {
                inner.is_compiling = true;
                inner.cleanup(device);
                log::debug!("queued Shader for recompilation {}", path.display());
                true
            }
            Err(err) => {
                log::error!("failed to queue Shader for recompilation: {err}");
                false
            }
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

    pub fn compile_code(&self) -> Result<(), anyhow::Error> {
        let result = self.compile_code_helper();
        let mut inner = self.inner.write().map_err(|_| anyhow::anyhow!("Lock poisoned"))?;
        inner.is_compiling = false;
        result
    }

    fn compile_code_helper(&self) -> Result<(), anyhow::Error> {
        // try not to panic in this function to keep the compile thread going

        let inner = self.inner.read().map_err(|_| anyhow::anyhow!("Lock poisoned"))?;
        let stage = inner.stage;
        let path = inner.path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Cannot compile a Shader without path"))?
            .clone();
        drop(inner); // do not keep the lock while compiling

        let code = ShaderInner::compile_code(stage, &path)?;
        let mut inner = self.inner.write().map_err(|_| anyhow::anyhow!("Lock poisoned"))?;
        inner.code = Some(code);
        inner.module = None;
        Ok(())
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
    path: Option<PathBuf>,
    code: Option<Box<[u32]>>,
    module: Option<vk::ShaderModule>,
    compile_sender: Option<Sender<Shader>>,
    is_compiling: bool,
    code_has_changed: bool,
}

impl ShaderInner {
    pub fn new(stage: ShaderStage) -> Self {
        Self {
            stage,
            path: None,
            code: None,
            module: None,
            compile_sender: None,
            is_compiling: false,
            code_has_changed: false,
        }
    }

    pub fn path<P: Into<PathBuf>>(mut self, path: P) -> Self {
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

    fn compile_code(stage: ShaderStage, path: &Path) -> Result<Box<[u32]>, anyhow::Error> {
        // try not to panic in this function to keep the compile thread going

        log::debug!("compiling Shader {} of stage {:?}", path.display(), stage);
        let source = std::fs::read_to_string(path)?.into();
        let compiler = Compiler::acquire()
            .ok_or_else(|| anyhow::anyhow!("Failed to acquire Compiler"))?;
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
            if let Some(path) = self.path.as_ref() {
                log::debug!("cleaning Shader {}", path.display());
            } else {
                log::debug!("cleaning Shader");
            }
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
        if !std::thread::panicking() &&  self.module.is_some() {
            log::error!("Shader was not cleaned up before beeing dropped");
        }
    }
}
