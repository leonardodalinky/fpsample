use std::{cell::Cell, ops::Index};

use numpy::ndarray::ArrayView1;

#[derive(Debug, Clone)]
pub struct Point<'v> {
    pub pos: ArrayView1<'v, f32>,
    pub dis: Cell<f32>,
}

impl<'v> Default for Point<'v> {
    fn default() -> Self {
        Self {
            pos: ArrayView1::from_shape(0, &[]).unwrap(),
            dis: Cell::new(f32::INFINITY),
        }
    }
}

impl<'v> Point<'v> {
    pub fn new(pos: ArrayView1<'v, f32>, dis: Option<f32>) -> Self {
        Self {
            pos,
            dis: Cell::new(dis.unwrap_or(f32::INFINITY)),
        }
    }

    #[inline]
    pub fn distance(&self, other: &Self) -> f32 {
        let dist = &self.pos - &other.pos;
        dist.mapv(|x| x.powi(2)).sum().sqrt()
    }

    #[inline]
    pub fn update_distance(&self, other: &Self) -> f32 {
        let t = f32::min(self.dis.get(), self.distance(other));
        self.dis.set(t);
        t
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.pos.len()
    }
}

impl<'v> Index<usize> for Point<'v> {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.pos.get(index).unwrap()
    }
}
