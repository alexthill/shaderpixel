use crate::obj::{Indices, Obj};

use std::num::NonZeroU32;

pub fn generate_env(start: [f32; 2], dims: [u32; 2], podests: &[[f32; 2]]) -> Obj {
    let mut vertices = Vec::new();
    for z in 0..dims[1] + 1 {
        for x in 0..dims[0] + 1 {
            vertices.push([start[0] + x as f32, 0.0, start[1] + z as f32]);
        }
    }

    let tex_coords = Vec::new();

    let mut faces = Vec::new();
    let w = dims[1] + 1;
    for z in 0..dims[1] {
        for x in 0..dims[0] {
            let vidx = x + z * w;
            faces.push(indices_to_face([vidx, vidx + w, vidx + 1 + w, vidx + 1]));
        }
    }

    for podest in podests {
        let vidx = vertices.len() as u32;
        for z in 0..2 {
            for x in 0..2 {
                vertices.push([podest[0] + x as f32, 0., podest[1] + z as f32]);
                vertices.push([podest[0] + x as f32, 0.99, podest[1] + z as f32]);
            }
        }
        faces.push(indices_to_face([vidx + 1, vidx + 5, vidx + 7, vidx + 3]));
        faces.push(indices_to_face([vidx + 0, vidx + 1, vidx + 3, vidx + 2]));
        faces.push(indices_to_face([vidx + 2, vidx + 3, vidx + 7, vidx + 6]));
        faces.push(indices_to_face([vidx + 6, vidx + 7, vidx + 5, vidx + 4]));
        faces.push(indices_to_face([vidx + 4, vidx + 5, vidx + 1, vidx + 0]));
    }

    Obj { vertices, tex_coords, faces }
}

fn indices_to_face(indices: [u32; 4]) -> ([Indices; 3], Option<Indices>) {
    let [a, b, c, d] = indices.map(|i| NonZeroU32::new(i + 1).unwrap());
    (
        [
            Indices { vertex: a, texture: None, normal: None },
            Indices { vertex: b, texture: None, normal: None },
            Indices { vertex: c, texture: None, normal: None },
        ],
        Some(Indices { vertex: d, texture: None, normal: None }),
    )
}
