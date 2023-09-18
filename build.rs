mod build_info {
    include!("build_info.rs");
}

fn main() {
    let mut b = cc::Build::new();
    b.file("src/bucket_fps/c_wrapper.cpp")
        .define(
            "BUCKET_FPS_MAX_DIM",
            build_info::BUCKET_FPS_MAX_DIM.to_string().as_str(),
        )
        .cpp(true)
        .std("c++14")
        .warnings(false)
        .include("src/bucket_fps/_ext");

    if let Ok(cpp_lib) = std::env::var("FORCE_CXXSTDLIB") {
        b.cpp_set_stdlib(cpp_lib.as_str());
    }

    b.compile("bucketfps");
}
