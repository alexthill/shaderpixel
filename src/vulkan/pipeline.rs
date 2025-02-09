use super::{
    geometry::Geometry,
    structs::{Shader, Vertex},
    swapchain::SwapchainProperties,
};

use ash::{vk, Device};
use std::{
    error::Error,
    ffi::CString,
};

pub struct Pipeline {
    pipeline_and_layout: Option<(vk::Pipeline, vk::PipelineLayout)>,
    pub geometry: Option<Geometry>,
    pub active: bool,
}

impl Pipeline {
    pub fn new(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        cull_mode: vk::CullModeFlags,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
        shaders: [&Shader; 2],
        geometry: Geometry,
    ) -> Self {
        let (pipeline, layout) = Self::create_pipeline(
            device,
            swapchain_properties,
            cull_mode,
            msaa_samples,
            render_pass,
            descriptor_set_layout,
            shaders,
        );

        Self {
            pipeline_and_layout: Some((pipeline, layout)),
            geometry: Some(geometry),
            active: true,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn recreate(
        &mut self,
        device: &Device,
        swapchain_properties: SwapchainProperties,
        cull_mode: vk::CullModeFlags,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
        shaders: [&Shader; 2],
    ) {
        if self.pipeline_and_layout.is_some() {
            panic!("Pipeline must be cleaned before recreation");
        }

        let (pipeline, layout) = Self::create_pipeline(
            device,
            swapchain_properties,
            cull_mode,
            msaa_samples,
            render_pass,
            descriptor_set_layout,
            shaders,
        );
        self.pipeline_and_layout = Some((pipeline, layout));
    }

    pub unsafe fn bind_to_cmd_buffer(
        &self,
        device: &Device,
        buffer: vk::CommandBuffer,
        descriptor_sets: &[vk::DescriptorSet],
    ) {
        let (pip_pip, pip_layout) = self.get().expect("pipeline must be initalized");
        device.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::GRAPHICS, pip_pip);
        let index_count = if let Some(geometry) = &self.geometry {
            let (vertex_buffer, index_buffer, index_count) = geometry.get().unwrap();
            device.cmd_bind_vertex_buffers(buffer, 0, &[vertex_buffer], &[0]);
            device.cmd_bind_index_buffer(buffer, index_buffer, 0, vk::IndexType::UINT32);
            index_count
        } else {
            0
        };
        device.cmd_bind_descriptor_sets(
            buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pip_layout,
            0,
            descriptor_sets,
            &[],
        );
        device.cmd_draw_indexed(buffer, index_count, 1, 0, 0, 0);
    }

    pub fn get(&self) -> Option<(vk::Pipeline, vk::PipelineLayout)> {
        self.pipeline_and_layout
    }

    pub unsafe fn cleanup(&mut self, device: &Device) {
        if let Some((pipeline, layout)) = self.pipeline_and_layout.take() {
            log::debug!("cleaning Pipeline");
            device.destroy_pipeline(pipeline, None);
            device.destroy_pipeline_layout(layout, None);
        }
    }

    fn create_shader_module(
        device: &Device,
        code: &[u32],
    ) -> Result<vk::ShaderModule, Box<dyn Error>> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);
        unsafe {
            Ok(device.create_shader_module(&create_info, None)?)
        }
    }

    fn create_pipeline(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        cull_mode: vk::CullModeFlags,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
        shaders: [&Shader; 2],
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let vertex_shader_module = Self::create_shader_module(device, shaders[0].data().unwrap())
            .expect("failed to load vertex shader spv file");
        let fragment_shader_module = Self::create_shader_module(device, shaders[1].data().unwrap())
            .expect("failed to load fragment shader spv file");

        let entry_point_name = CString::new("main").unwrap();
        let vertex_shader_state_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&entry_point_name);
        let fragment_shader_state_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&entry_point_name);
        let shader_states_infos = [vertex_shader_state_info, fragment_shader_state_info];

        let vertex_binding_descs = [Vertex::get_binding_description()];
        let vertex_attribute_descs = Vertex::get_attribute_descriptions();
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_binding_descs)
            .vertex_attribute_descriptions(&vertex_attribute_descs);

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_properties.extent.width as _,
            height: swapchain_properties.extent.height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let viewports = [viewport];
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_properties.extent,
        };
        let scissors = [scissor];
        let viewport_info = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(cull_mode)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0);

        let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(msaa_samples)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false)
            .front(Default::default())
            .back(Default::default());

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);
        let color_blend_attachments = [color_blend_attachment];

        let color_blending_info = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let layout = {
            let layouts = [descriptor_set_layout];
            let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
            unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_states_infos)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampling_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blending_info)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        let pipeline_infos = [pipeline_info];

        let pipeline = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        };

        (pipeline, layout)
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if self.pipeline_and_layout.is_some() {
            panic!("Pipeline was not cleaned up before beeing dropped");
        }
    }
}
