use super::buffer;
use super::context::VkContext;
use super::vertex::Vertex;

use ash::{vk, Device};
use std::rc::Rc;

#[derive(Clone)]
pub struct Geometry {
    rc: Option<Rc<()>>,
    vertex_binding_description: vk::VertexInputBindingDescription,
    vertex_attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    index_count: u32,
}

impl Geometry {
    pub fn new<V: Vertex + Copy>(
        vk_context: &VkContext,
        transient_command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        vertices: &[V],
        indices: &[u32],
    ) -> Self {
        let (vertex_buffer, vertex_buffer_memory) = Self::create_buffer_with_data::<u32, _>(
            vk_context,
            transient_command_pool,
            graphics_queue,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vertices,
        );
        let (index_buffer, index_buffer_memory) = Self::create_buffer_with_data::<u16, _>(
            vk_context,
            transient_command_pool,
            graphics_queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            indices,
        );

        Self {
            rc: Some(Rc::new(())),
            vertex_binding_description: V::get_binding_description(),
            vertex_attribute_descriptions: V::get_attribute_descriptions(),
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            index_count: indices.len() as _,
        }
    }

    pub fn get(&self) -> Option<(vk::Buffer, vk::Buffer, u32)> {
        self.rc.as_ref().map(|_| (self.vertex_buffer, self.index_buffer, self.index_count))
    }

    pub fn get_binding_description(&self) -> vk::VertexInputBindingDescription {
        self.vertex_binding_description
    }

    pub fn get_attribute_descriptions(&self) -> &[vk::VertexInputAttributeDescription] {
        &self.vertex_attribute_descriptions
    }

    pub unsafe fn cleanup(mut self, device: &Device) {
        if self.rc.take().map(|rc| Rc::strong_count(&rc) == 1).unwrap_or(false) {
            log::debug!("cleaning Geometry");
            device.free_memory(self.index_buffer_memory, None);
            device.destroy_buffer(self.index_buffer, None);
            device.free_memory(self.vertex_buffer_memory, None);
            device.destroy_buffer(self.vertex_buffer, None);
        }
    }

    /// Create a buffer and its gpu memory and fill it.
    ///
    /// This function internally creates an host visible staging buffer and
    /// a device local buffer. The data is first copied from the cpu to the
    /// staging buffer. Then we copy the data from the staging buffer to the
    /// final buffer using a one-time command buffer.
    fn create_buffer_with_data<A, T: Copy>(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let device = vk_context.device();
        let size = size_of_val(data) as vk::DeviceSize;
        let (staging_buffer, staging_memory, staging_mem_size) = buffer::create_buffer(
            vk_context,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align = ash::util::Align::new(data_ptr, align_of::<A>() as _, staging_mem_size);
            align.copy_from_slice(data);
            device.unmap_memory(staging_memory);
        };

        let (buffer, memory, _) = buffer::create_buffer(
            vk_context,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        buffer::copy_buffer(
            device,
            command_pool,
            transfer_queue,
            staging_buffer,
            buffer,
            size,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        };

        (buffer, memory)
    }
}

impl Drop for Geometry {
    fn drop(&mut self) {
        if !std::thread::panicking() && self.rc.is_some() {
            log::error!("Geometry was not cleaned up before beeing dropped");
        }
    }
}
