#include "./vec4.ocl"
#include "./ray.ocl"

float hit_sphere(const float4 center,  float radius, const ray r) {
    float4 oc = r.orig - center;
    float a = vec4_length_squared(r.dir);
    float half_b = vec4_dot(oc, r.dir);
    float c = vec4_length_squared(oc) - radius*radius;
    float discriminant = half_b*half_b - a*c;

    if (discriminant < 0) {
        return -1.0;
    } else {
        return (0.0f-half_b - sqrt(discriminant) ) / a;
    }
}

float4 ray_color(const ray r) {
    float t = hit_sphere((float4)(0.0f, 0.0f, -1.0f, 0.0f), 0.5f, r);
    //printf("t = %f\n", t);
    if (t > 0.0f) {
        float4 N = unit_vector(ray_at(r, t) - (float4)(0.0f, 0.0f, -1.0f, 0.0f));
        return 0.5f*(float4)(N.x+1.0f, N.y+1.0f, N.z+1.0f, 0.0f);
    }
    float4 unit_direction = unit_vector(r.dir);
    t = 0.5f*(unit_direction.y + 1.0f);
    return (1.0f-t)*(float4)(1.0f, 1.0f, 1.0f, 0.0f) + t*(float4)(0.5f, 0.7f, 1.0f, 0.0f);
}

__kernel void engine(
     int image_width,
     int image_hieght,
     __global unsigned char * image_buffer
){
    int gid = get_global_id(0)-image_width;
    gid  = abs(gid);
   if(gid<0){return;}

  //printf("gid = %d", gid);

  // Image
    const float aspect_ratio = image_width/image_hieght;
    //const int image_width = 400;
    //const int image_height = (int)(image_width / aspect_ratio);

    // Camera

    float viewport_height = 2.0f;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0f;

    float4 origin = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 horizontal = (float4)(viewport_width, 0.0f, 0.0f, 0.0f);
    float4 vertical = (float4)(0.0f, viewport_height, 0.0f, 0.0f);
    float4 lower_left_corner = origin - horizontal/2 - vertical/2 - (float4)(0.0f, 0.0f, focal_length, 0.0f);


    for (int i = 0; i < image_hieght ; i++) {

          float v = (float)(gid) / (float)(image_width-1);
          float u = (float)(i) / (float)(image_hieght-1);
          //printf("i_h = %d, i_w = %d, i = %d, gid = %d, r = %f, g = %f\n",image_height, image_width, i, gid, r, g);
          ray r = {origin, lower_left_corner + u*horizontal + v*vertical - origin};
          float4 pixel_color = ray_color(r);

          //float b = 0.25;

          image_buffer[gid*image_hieght*3+i*3] = 255 * pixel_color.x;
          image_buffer[gid*image_hieght*3+i*3+1] = 255 * pixel_color.y;
          image_buffer[gid*image_hieght*3+i*3+2] = 255 * pixel_color.z;
            
            //printf("%u %u %u\n", 255 * r, 255 * g, 255 * b);
       }
}