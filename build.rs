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
        // .std("c++14")    FIXME: error in some arch
        .warnings(false)
        .include("src/bucket_fps/_ext")
        .compile("bucketfps");
}
