use shaderpixel::{
    env_generator::default_env,
    fs::Carousel,
    math::{Deg, Matrix4, Vector3, Vector4},
    vulkan::{Shader, Shaders, ShaderArt, ShaderInner, VkApp},
};

use anyhow::Context;
use glslang::ShaderStage;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};
use std::{
    path::Path,
    time::Instant,
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const TITLE: &str = "shaderpixel";
const START_POSITION: Vector3 = Vector3::new_init([0., 1.5, 3.]);
const TEXTURE_WEIGHT_CHANGE_SPEED: f32 = 0.5; // change will take 2 secs from 0 to 1

fn check_if_image(path: &Path) -> bool {
    path.extension().map(|ext| ext == "jpg" || ext == "png").unwrap_or_default()
}

fn main() {
    println!("Usage:");
    println!("Run with RUST_LOG=debug to see logging output");
    println!();
    println!("Right-Click: rotate camera with mouse");
    println!("Mouse-Wheel: change movement speed");
    println!("WASD: move around");
    println!("Space and Left-Shift: move up and down");
    println!("Left-Ctrl: enter fly mode");
    println!("Right-Ctrl: hot reload shaders");
    println!("B: toggle skybox");
    println!("R: reset camera and object");
    println!();

    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        position: START_POSITION,
        ..Default::default()
    };
    app.image_carousel.set_dir("assets/images");
    event_loop.run_app(&mut app).unwrap();
}

#[derive(Default)]
pub struct KeyStates {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
}

#[derive(Default)]
struct App {
    window: Option<Window>,
    vulkan: Option<VkApp>,

    fps: Option<(Instant, u32)>,
    last_frame: Option<Instant>,
    time: f32, // time passed since app start in seconds

    pressed: KeyStates,
    load_next_image: bool,
    reload_shaders: bool,
    is_right_clicked: bool,
    cursor_position: Option<[i32; 2]>,
    cursor_delta: [i32; 2],
    tex_weight_change: f32,
    is_fullscreen: bool,
    scroll_lines: f32,

    angle_yaw: Deg<f32>,
    angle_pitch: Deg<f32>,
    position: Vector3,
    fly_mode: bool,

    image_carousel: Carousel,
}

impl App {
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<(), anyhow::Error> {
        let window_attrs = Window::default_attributes()
            .with_title(TITLE)
            .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT));
        let window = event_loop.create_window(window_attrs).context("Failed to create window")?;

        let nobj = default_env().normalize()?;
        //let nobj = NormalizedObj::from_reader(fs::load("assets/models/env.obj")?)?;
        let image_path = self.image_carousel.get_next(0, check_if_image)
            .context("Failed to find an image")?;
        let dims = [WIDTH, HEIGHT];

        let vert_shader_art2d: Shader = ShaderInner::new(ShaderStage::Vertex)
            .path("assets/shaders/art2d.vert").into();
        let vert_shader_art3d: Shader = ShaderInner::new(ShaderStage::Vertex)
            .path("assets/shaders/art3d.vert").into();
        let shaders = Shaders {
            main_vert: ShaderInner::new(ShaderStage::Vertex)
                .bytes(include_bytes!(concat!(env!("OUT_DIR"), "/shader.vert.spv")))?.into(),
            main_frag: ShaderInner::new(ShaderStage::Fragment)
                .bytes(include_bytes!(concat!(env!("OUT_DIR"), "/shader.frag.spv")))?.into(),
            cube_vert: ShaderInner::new(ShaderStage::Vertex)
                .bytes(include_bytes!(concat!(env!("OUT_DIR"), "/cubemap.vert.spv")))?.into(),
            cube_frag: ShaderInner::new(ShaderStage::Fragment)
                .bytes(include_bytes!(concat!(env!("OUT_DIR"), "/cubemap.frag.spv")))?.into(),
            // draw 2D art before 3D so that it can be seen through transparent stuff
            shaders_art: vec![
                ShaderArt {
                    name: "Mandelbrot".to_owned(),
                    is_3d: false,
                    vert: vert_shader_art2d.clone(),
                    frag: ShaderInner::new(ShaderStage::Fragment)
                        .path("assets/shaders/mandelbrot.frag").into(),
                    model_matrix: Matrix4::from_translation([5.99, 1.5, -1.5].into())
                        * Matrix4::from_scale(0.5)
                        * Matrix4::from_angle_y(Deg(90.)),
                },
                ShaderArt {
                    name: "Sdf Cat".to_owned(),
                    is_3d: false,
                    vert: vert_shader_art2d,
                    frag: ShaderInner::new(ShaderStage::Fragment)
                        .path("assets/shaders/cat.frag").into(),
                    model_matrix: Matrix4::from_translation([5.99, 1.5, -4.5].into())
                        * Matrix4::from_scale(0.5)
                        * Matrix4::from_angle_y(Deg(90.)),
                },
                ShaderArt {
                    name: "Mandelbox".to_owned(),
                    is_3d: true,
                    vert: vert_shader_art3d.clone(),
                    frag: ShaderInner::new(ShaderStage::Fragment)
                        .path("assets/shaders/mandelbox.frag").into(),
                    model_matrix: Matrix4::from_translation([-2.5, 1.51, -0.5].into())
                        * Matrix4::from_scale(0.5),
                },
                ShaderArt {
                    name: "Menger Sponge".to_owned(),
                    is_3d: true,
                    vert: vert_shader_art3d.clone(),
                    frag: ShaderInner::new(ShaderStage::Fragment)
                        .path("assets/shaders/mengersponge.frag").into(),
                    model_matrix: Matrix4::from_translation([2.5, 1.51, -0.5].into())
                        * Matrix4::from_scale(0.5),
                },
                ShaderArt {
                    name: "Solar".to_owned(),
                    is_3d: true,
                    vert: vert_shader_art3d.clone(),
                    frag: ShaderInner::new(ShaderStage::Fragment)
                        .path("assets/shaders/solar.frag").into(),
                    model_matrix: Matrix4::from_translation([-2.5, 1.51, -5.5].into())
                        * Matrix4::from_scale(0.5),
                },
                ShaderArt {
                    name: "Mountain".to_owned(),
                    is_3d: true,
                    vert: vert_shader_art3d,
                    frag: ShaderInner::new(ShaderStage::Fragment)
                        .path("assets/shaders/mountain.frag").into(),
                    model_matrix: Matrix4::from_translation([2.5, 1.51, -5.5].into())
                        * Matrix4::from_scale(0.5),
                },
            ],
        };

        let vulkan = VkApp::new(
            &window,
            dims,
            &image_path,
            nobj,
            shaders,
        )?;

        self.vulkan = Some(vulkan);
        self.window = Some(window);
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Err(err) = self.init(event_loop) {
            log::error!("Error while starting: {err}");
            log::error!("{err:#?}");
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        logical_key,
                        physical_key: PhysicalKey::Code(physical_key_code),
                        repeat: false,
                        ..
                    },
                ..
            } => {
                let pressed = state.is_pressed();
                match physical_key_code {
                    KeyCode::KeyW => self.pressed.forward = pressed,
                    KeyCode::KeyA => self.pressed.left = pressed,
                    KeyCode::KeyS => self.pressed.backward = pressed,
                    KeyCode::KeyD => self.pressed.right = pressed,
                    KeyCode::Space => self.pressed.up = pressed,
                    KeyCode::ShiftLeft => self.pressed.down = pressed,
                    KeyCode::ControlRight if pressed => self.reload_shaders = true,
                    KeyCode::ControlLeft if pressed => self.fly_mode = !self.fly_mode,
                    _ => {}
                }

                let Some(vulkan) = self.vulkan.as_mut() else { return };
                match (logical_key.as_ref(), pressed) {
                    (Key::Character("b"), true) => {
                        vulkan.toggle_cubemap();
                        vulkan.dirty_swapchain = true;
                    }
                    (Key::Character("f"), true) => {
                        let fullscreen = if self.is_fullscreen {
                            None
                        } else {
                            Some(Fullscreen::Borderless(None))
                        };
                        self.window.as_mut().unwrap().set_fullscreen(fullscreen);
                        self.is_fullscreen = !self.is_fullscreen;
                    }
                    (Key::Character("i"), true) => {
                        self.load_next_image = true;
                        if vulkan.texture_weight == 0. || self.tex_weight_change < 0. {
                            self.tex_weight_change = TEXTURE_WEIGHT_CHANGE_SPEED;
                        }
                    }
                    (Key::Character("l"), true) => {
                        vulkan.reset_ubo();
                        self.angle_yaw = Default::default();
                        self.angle_pitch = Default::default();
                        self.position = START_POSITION;
                        self.scroll_lines = 0.0;
                    }
                    (Key::Character("t"), true) => {
                        self.tex_weight_change = if self.tex_weight_change == 0. {
                            TEXTURE_WEIGHT_CHANGE_SPEED
                        } else {
                            -self.tex_weight_change
                        };
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized { .. } => {
                self.vulkan.as_mut().unwrap().dirty_swapchain = true;
            }
            WindowEvent::MouseInput { button: MouseButton::Right, state, .. } => {
                self.is_right_clicked = state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos: (i32, i32) = position.into();
                if self.is_right_clicked {
                    if let Some(old_pos) = self.cursor_position {
                        self.cursor_delta[0] += new_pos.0 - old_pos[0];
                        self.cursor_delta[1] += new_pos.1 - old_pos[1];
                    }
                }
                self.cursor_position = Some([new_pos.0, new_pos.1]);
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, v_lines),
                ..
            } => {
                self.scroll_lines += v_lines;
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if event_loop.exiting() {
            return;
        }

        if let Some((start, count)) = self.fps.as_mut() {
            let time = start.elapsed();
            *count += 1;
            if time.as_millis() > 1000 {
                use std::io::Write;

                eprint!("fps: {}        \r", *count as f32 / time.as_secs_f32());
                std::io::stdout().flush().unwrap();
                *start = Instant::now();
                *count = 0;
            }
        } else {
            self.fps = Some((Instant::now(), 0));
        }

        let app = self.vulkan.as_mut().unwrap();
        let window = self.window.as_ref().unwrap();

        if app.dirty_swapchain {
            let size = window.inner_size();
            if size.width > 0 && size.height > 0 {
                app.recreate_swapchain(size.width, size.height);
            } else {
                return;
            }
        }

        let elapsed = self.last_frame.map(|instant| instant.elapsed()).unwrap_or_default();
        let delta = elapsed.as_secs_f32() * (self.scroll_lines * 0.4).exp();
        self.last_frame = Some(Instant::now());
        self.time += elapsed.as_secs_f32();

        let extent = window.inner_size();
        let x_ratio = self.cursor_delta[0] as f32 / extent.width as f32;
        let y_ratio = self.cursor_delta[1] as f32 / extent.height as f32;

        if self.is_right_clicked {
            self.angle_yaw += Deg(x_ratio * 180.);
            self.angle_pitch += Deg(y_ratio * 180.);
        }
        self.cursor_delta = [0, 0];

        let translation = Vector4::from([
            (self.pressed.left    as i8 - self.pressed.right    as i8) as f32,
            (self.pressed.down    as i8 - self.pressed.up       as i8) as f32,
            (self.pressed.forward as i8 - self.pressed.backward as i8) as f32,
            0.,
        ]) * delta * 2.;
        let rot = if self.fly_mode {
            Matrix4::from_angle_y(-self.angle_yaw) * Matrix4::from_angle_x(-self.angle_pitch)
        } else {
            Matrix4::from_angle_y(-self.angle_yaw)
        };
        self.position += (-translation * rot).resize();

        app.view_matrix = Matrix4::from_angle_x(self.angle_pitch)
            * Matrix4::from_angle_y(self.angle_yaw)
            * Matrix4::from_translation(-self.position);

        if self.load_next_image {
            match self.image_carousel.get_next(1, check_if_image) {
                Ok(path) => {
                    if let Err(err) = app.load_new_texture(&path) {
                        log::warn!("Error while loading new image: {err}");
                        log::warn!("{err:#?}");
                    }
                }
                Err(err) => log::warn!("Failed to find an image: {err}"),
            };
            self.load_next_image = false;
        }
        if self.reload_shaders {
            app.reload_shaders();
            self.reload_shaders = false;
        }

        app.texture_weight = (app.texture_weight + self.tex_weight_change * delta).clamp(0., 1.);

        app.dirty_swapchain = app.draw_frame(self.time);
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        if let Some(vulkan) = self.vulkan.as_ref() {
            vulkan.wait_gpu_idle();
        }
    }
}
