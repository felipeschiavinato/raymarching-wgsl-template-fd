const THREAD_COUNT = 16;
const PI = 3.1415927f;
const MAX_DIST = 1000.0;

@group(0) @binding(0)  
  var<storage, read_write> fb : array<vec4f>;

@group(1) @binding(0)
  var<storage, read_write> uniforms : array<f32>;

@group(2) @binding(0)
  var<storage, read_write> shapesb : array<shape>;

@group(2) @binding(1)
  var<storage, read_write> shapesinfob : array<vec4f>;

struct shape {
  transform : vec4f, // xyz = position
  radius : vec4f, // xyz = scale, w = global scale
  rotation : vec4f, // xyz = rotation
  op : vec4f, // x = operation, y = k value, z = repeat mode, w = repeat offset
  color : vec4f, // xyz = color
  animate_transform : vec4f, // xyz = animate position value (sin amplitude), w = animate speed
  animate_rotation : vec4f, // xyz = animate rotation value (sin amplitude), w = animate speed
  quat : vec4f, // xyzw = quaternion
  transform_animated : vec4f, // xyz = position buffer
};

struct march_output {
  color : vec3f,
  depth : f32,
  outline : bool,
};

// Smooth Union
fn op_smooth_union(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f {
  var k_2 = max(k, 0.001);
  let h = clamp(0.5 + 0.5 * (d2 - d1) / k_2, 0.0, 1.0);
  let d = mix(d2, d1, h) - k * h * (1.0 - h);
  let col = mix(col2, col1, h);
  return vec4f(col, d);
}

fn op_smooth_subtraction(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f {
  var k_2 = max(k, 0.001);
  let h = clamp(0.5 - 0.5 * (d2 + d1) / k_2, 0.0, 1.0);
  let d = mix(d2, -d1, h) + k * h * (1.0 - h);
  let col = mix(col2, col1, h);
  return vec4f(col, d);
}

// Smooth Intersection
fn op_smooth_intersection(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f {
  var k_2 = max(k, 0.001);
  let h = clamp(0.5 - 0.5 * (d2 - d1) / k_2, 0.0, 1.0);
  let d = mix(d2, d1, h) + k * h * (1.0 - h);
  let col = mix(col2, col1, h);
  return vec4f(col, d);
}

// CSG Operations
fn op(op_type: f32, d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f {
  // Union
  if (op_type < 1.0) {
    return op_smooth_union(d1, d2, col1, col2, k);
  }
  // Subtraction
  if (op_type < 2.0) {
    return op_smooth_subtraction(d2, d1, col2, col1, k);
  }
  // Intersection
  return op_smooth_intersection(d2, d1, col2, col1, k);
}

// Repeat Function
fn repeat(p: vec3f, offset: vec3f) -> vec3f {
  return p - offset * floor(p / offset + 0.5);
}

// Transform Point with Options
fn transform_p(p: vec3f, option: vec2f) -> vec3f {
  if (option.x <= 1.0) {
    return p;
  }
  return repeat(p, vec3f(option.y));
}


fn scene(p: vec3f) -> vec4f // xyz = color, w = distance
{
    var d = mix(100.0, p.y, uniforms[17]);

    var spheresCount = i32(uniforms[2]);
    var boxesCount = i32(uniforms[3]);
    var torusCount = i32(uniforms[4]);

    var all_objects_count = spheresCount + boxesCount + torusCount;
    var result = vec4f(vec3f(1.0), d);

    for (var i = 0; i < all_objects_count; i = i + 1)
    {

        let shape_info = shapesinfob[i];
        let shape_type = shape_info.x;
        let shape_index = i32(shape_info.y);
        let shape = shapesb[shape_index];
        
        let animated_transform = animate(shape.animate_transform.xyz, shape.animate_transform.w, 0.0);
        let animated_rotation = animate(shape.animate_rotation.xyz, shape.animate_rotation.w, 0.0);

        let quaternion = quaternion_from_euler(animated_rotation + shape.rotation.xyz);


        var p_transformed = p - (shape.transform.xyz + animated_transform);
        p_transformed = transform_p(p_transformed, shape.op.zw);

        
        var dist: f32;
        let color = shape.color.xyz;
        
        if (shape_type < 1.0)
        {
          dist = sdf_sphere(p_transformed, shape.radius, quaternion);
        } else if (shape_type < 2.0)   {
          dist = sdf_round_box(p_transformed, shape.radius.xyz, shape.radius.w, quaternion);
        } else if (shape_type < 3.0)  {
          dist = sdf_torus(p_transformed, shape.radius.xy,  quaternion);
        } else {
          dist = MAX_DIST;
        }

        var aux = vec4f(shape.color.xyz, dist);
        result = op(shape.op.x, result.w, aux.w, result.xyz, color, shape.op.y);
    }

    return result;
}

fn march(ro: vec3f, rd: vec3f) -> march_output
{
  var max_marching_steps = i32(uniforms[5]);
  var EPSILON = uniforms[23];

  var depth = 0.0;
  var color = vec3f(1.0);
  var march_step = uniforms[22];
  
  for (var i = 0; i < max_marching_steps; i = i + 1)
  {
      let p = ro + depth * rd;
      let res = scene(p);
      
      if (res.w < EPSILON) {
          color = res.xyz;
          return march_output(color, depth, false);
      }
      
      if (depth > MAX_DIST) {
          return march_output(color, depth, false);
      }
      
      depth += res.w * march_step;
  }

  return march_output(color, depth, false);
}

fn get_normal(p: vec3f) -> vec3f
{
  let EPSILON = uniforms[23];
  let dx = vec3f(EPSILON, 0.0, 0.0);
  let dy = vec3f(0.0, EPSILON, 0.0);
  let dz = vec3f(0.0, 0.0, EPSILON);

  let n = vec3f(
    scene(p + dx).w - scene(p - dx).w,
    scene(p + dy).w - scene(p - dy).w,
    scene(p + dz).w - scene(p - dz).w
  );

  return normalize(n);
}

// https://iquilezles.org/articles/rmshadows/
fn get_soft_shadow(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, k: f32) -> f32
{
  var res = 1.0;
  var t = tmin;
  var EPSILON = uniforms[23];

  for (var i = 0; i < 256; i = i + 1) {
    let h = scene(ro + rd * t).w;
    if (h < EPSILON) {
      return 0.0;
    }
    res = min(res, k * h / t);
    t += h;

    if (t >= tmax) {
      break;
    }
  }

  return res;
}

fn get_AO(current: vec3f, normal: vec3f) -> f32
{
  var occ = 0.0;
  var sca = 1.0;
  for (var i = 0; i < 5; i = i + 1)
  {
    var h = 0.001 + 0.15 * f32(i) / 4.0;
    var d = scene(current + h * normal).w;
    occ += (h - d) * sca;
    sca *= 0.95;
  }

  return clamp( 1.0 - 2.0 * occ, 0.0, 1.0 ) * (0.5 + 0.5 * normal.y);
}

fn get_ambient_light(light_pos: vec3f, sun_color: vec3f, rd: vec3f) -> vec3f
{
  var backgroundcolor1 = int_to_rgb(i32(uniforms[12]));
  var backgroundcolor2 = int_to_rgb(i32(uniforms[29]));
  var backgroundcolor3 = int_to_rgb(i32(uniforms[30]));
  
  var ambient = backgroundcolor1 - rd.y * rd.y * 0.5;
  ambient = mix(ambient, 0.85 * backgroundcolor2, pow(1.0 - max(rd.y, 0.0), 4.0));

  var sundot = clamp(dot(rd, normalize(vec3f(light_pos))), 0.0, 1.0);
  var sun = 0.25 * sun_color * pow(sundot, 5.0) + 0.25 * vec3f(1.0,0.8,0.6) * pow(sundot, 64.0) + 0.2 * vec3f(1.0,0.8,0.6) * pow(sundot, 512.0);
  ambient += sun;
  ambient = mix(ambient, 0.68 * backgroundcolor3, pow(1.0 - max(rd.y, 0.0), 16.0));

  return ambient;
}

fn get_light(current: vec3f, obj_color: vec3f, rd: vec3f) -> vec3f
{
  var light_position = vec3f(uniforms[13], uniforms[14], uniforms[15]);
  var sun_color = int_to_rgb(i32(uniforms[16]));
  var ambient = get_ambient_light(light_position, sun_color, rd);
  var normal = get_normal(current);

  // calculate light based on the normal
  // if the object is too far away from the light source, return ambient light
  if (length(current) > uniforms[20] + uniforms[8])
  {
    return ambient;
  }

  let light_dir = normalize(light_position - current);
  var diffuse = max(dot(normal, light_dir), 0.0);

  // Shadows
  let shadow = get_soft_shadow(current, light_dir, 0.1, uniforms[25], 1/uniforms[21]);
  diffuse *= shadow;

  // Ambient Occlusion
  let ao = get_AO(current, normal);

  // Combine
  let color = obj_color * diffuse * sun_color * ao + ambient * obj_color * ao;

  return color;
}

fn set_camera(ro: vec3f, ta: vec3f, cr: f32) -> mat3x3<f32>
{
  var cw = normalize(ta - ro);
  var cp = vec3f(sin(cr), cos(cr), 0.0);
  var cu = normalize(cross(cw, cp));
  var cv = normalize(cross(cu, cw));
  return mat3x3<f32>(cu, cv, cw);
}

fn animate(val: vec3f, time_scale: f32, offset: f32) -> vec3f
{
    var time = uniforms[0];

    let x = val.x * -cos(time * time_scale + offset);
    let y = val.y * sin(time * time_scale + offset);
    let z = val.z * sin(time * time_scale + offset);

    // Return the result as a vec3f
    return vec3f(x, y, z);
}

@compute @workgroup_size(THREAD_COUNT, 1, 1)
fn preprocess(@builtin(global_invocation_id) id : vec3u)
{
  var time = uniforms[0];
  var spheresCount = i32(uniforms[2]);
  var boxesCount = i32(uniforms[3]);
  var torusCount = i32(uniforms[4]);
  var all_objects_count = spheresCount + boxesCount + torusCount;

  if (id.x >= u32(all_objects_count))
  {
    return;
  }

  // optional: performance boost
  // Do all the transformations here and store them in the buffer since this is called only once per object and not per pixel
}

@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id : vec3u)
{
  // unpack data
  var fragCoord = vec2f(f32(id.x), f32(id.y));
  var rez = vec2(uniforms[1]);
  var time = uniforms[0];

  // camera setup
  var lookfrom = vec3(uniforms[6], uniforms[7], uniforms[8]);
  var lookat = vec3(uniforms[9], uniforms[10], uniforms[11]);
  var camera = set_camera(lookfrom, lookat, 0.0);
  var ro = lookfrom;

  // get ray direction
  var uv = (fragCoord - 0.5 * rez) / rez.y;
  uv.y = -uv.y;
  var rd = camera * normalize(vec3(uv, 1.0));

  // Call march function and get the color/depth
  let res = march(ro, rd);

  // If hit
  var color: vec3f;
  if (res.depth < uniforms[20]) {
    let current = ro + rd * res.depth;
    color = get_light(current, res.color, rd);
  } else {
    // Background color
    color = get_ambient_light(vec3f(uniforms[13], uniforms[14], uniforms[15]), int_to_rgb(i32(uniforms[16])), rd);
  }
  
  // display the result
  color = linear_to_gamma(color);
  fb[mapfb(id.xy, uniforms[1])] = vec4(color, 1.0);
}