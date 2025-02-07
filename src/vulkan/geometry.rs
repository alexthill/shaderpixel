use super::buffer;
use super::context::VkContext;
use super::structs::Vertex;

use ash::{vk, Device};

#[derive(Copy, Clone)]
pub struct Geometry {
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    pub index_count: usize,
}

impl Geometry {
    pub fn new(
        vk_context: &VkContext,
        transient_command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        vertices: &[Vertex],
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
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            index_count: indices.len(),
        }
    }

    pub unsafe fn cleanup(self, device: &Device) {
        device.free_memory(self.index_buffer_memory, None);
        device.destroy_buffer(self.index_buffer, None);
        device.free_memory(self.vertex_buffer_memory, None);
        device.destroy_buffer(self.vertex_buffer, None);
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
