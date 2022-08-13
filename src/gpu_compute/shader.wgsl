@group(0)
@binding(0)
var<storage, read_write> v_indices: array<u32>; // this is used as both input and output for convenience

fn testing(n_base: u32) -> u32{
    var n: u32 = n_base;
    var i: u32 = n * n;
    return i;
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_indices[global_id.x] = testing(v_indices[global_id.x]);
}