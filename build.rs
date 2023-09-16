mod build_info {
    include!("build_info.rs");
}

fn main() {
    cc::Build::new()
        .file("src/bucket_fps/c_wrapper.cpp")
        .define(
            "BUCKET_FPS_MAX_DIM",
            build_info::BUCKET_FPS_MAX_DIM.to_string().as_str(),
        )
        .cpp(true)
        .flag("-Wno-array-bounds")
        .flag("-std=c++17")
        .include("src/bucket_fps/_ext")
        .compile("bucketfps");
}
