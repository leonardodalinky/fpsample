use cty;

extern "C" {
    pub(super) fn bucket_fps_kdtree(
        raw_data: *const cty::c_float,
        n_points: cty::size_t,
        dim: cty::size_t,
        n_samples: cty::size_t,
        start_idx: cty::size_t,
        sampled_point_indices: *mut cty::size_t,
    ) -> cty::c_int;

    pub(super) fn bucket_fps_kdline(
        raw_data: *const cty::c_float,
        n_points: cty::size_t,
        dim: cty::size_t,
        n_samples: cty::size_t,
        start_idx: cty::size_t,
        height: cty::size_t,
        sampled_point_indices: *mut cty::size_t,
    ) -> cty::c_int;
}
