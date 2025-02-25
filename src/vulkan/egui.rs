use super::{
    context::VkContext,
};

use ash::{vk, Instance};
use egui::{ClippedPrimitive, Context, TextureId, ViewportId};
use egui_ash_renderer::Renderer;
use egui_winit::State;
use raw_window_handle::HasDisplayHandle;
use winit::window::Window;

pub struct Egui {
    renderer: Renderer,
    context: Context,
    state: State,
    textures_to_free: Option<Vec<TextureId>>,
    pixels_per_point: f32,
    clipped_primitives: Vec<ClippedPrimitive>,
}

impl Egui {
    pub fn new(
        instance: &Instance,
        vk_context: &VkContext,
        render_pass: vk::RenderPass,
        display_target: &dyn HasDisplayHandle,
        in_flight_frames: usize,
    ) -> Result<Self, anyhow::Error> {
        let renderer = Renderer::with_default_allocator(
            instance,
            vk_context.physical_device(),
            vk_context.device().clone(),
            render_pass,
            egui_ash_renderer::Options {
                in_flight_frames,
                srgb_framebuffer: false,
                ..Default::default()
            },
        )?;
        let context = Context::default();
        let state = State::new(
            context.clone(),
            ViewportId::ROOT,
            display_target,
            None,
            None,
            None,
        );

        Ok(Self {
            renderer,
            context,
            state,
            textures_to_free: None,
            pixels_per_point: 0.0,
            clipped_primitives: Vec::new(),
        })
    }

    pub fn set_render_pass(&mut self, render_pass: vk::RenderPass) {
        self.renderer.set_render_pass(render_pass)
            .expect("Failed to rebuild egui renderer pipeline");
    }

    pub fn prepare_draw(
        &mut self,
        window: &Window,
        graphics_queue: vk::Queue,
        command_pool: vk::CommandPool,
    ) {
        if let Some(textures) = self.textures_to_free.take() {
            self.renderer.free_textures(&textures)
                .expect("Failed to free textures");
        }

        let raw_input = self.state.take_egui_input(&window);
        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            ..
        } = self.context.run(raw_input, |ctx| {
            Self::desktop_ui(ctx);
        });
        self.state.handle_platform_output(&window, platform_output);
        self.pixels_per_point = pixels_per_point;

        if !textures_delta.free.is_empty() {
            self.textures_to_free = Some(textures_delta.free.clone());
        }
        if !textures_delta.set.is_empty() {
            self.renderer.set_textures(
                graphics_queue,
                command_pool,
                textures_delta.set.as_slice(),
            ).expect("Failed to update texture");
        }

        self.clipped_primitives = self.context.tessellate(shapes, pixels_per_point);
    }

    pub fn draw(&mut self, buffer: vk::CommandBuffer, extent: vk::Extent2D) -> Result<(), anyhow::Error> {
        Ok(self.renderer.cmd_draw(
            buffer,
            extent,
            self.pixels_per_point,
            &self.clipped_primitives,
        )?)
    }

    fn desktop_ui(ctx: &Context) {
        egui::SidePanel::right("egui_demo_panel")
            .resizable(false)
            .default_width(160.0)
            .min_width(160.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.vertical_centered(|ui| {
                    ui.heading("✒ egui demo");
                });

                ui.separator();

                use egui::special_emojis::GITHUB;
                ui.hyperlink_to(
                    format!("{GITHUB} egui on GitHub"),
                    "https://github.com/emilk/egui",
                );
                ui.hyperlink_to(
                    "@ernerfeldt.bsky.social",
                    "https://bsky.app/profile/ernerfeldt.bsky.social",
                );

                ui.separator();

                if ui.button("Click me").clicked() {
                    log::info!("button clicked");
                }
            });
    }
}
