use std::{
    cell::{Cell, RefCell},
    ops::Range,
    rc::Rc,
};

use super::point::Point;

pub(super) struct KDNode<'v> {
    pub points: Rc<RefCell<Vec<Point<'v>>>>,
    pub point_left: usize,
    pub point_right: usize,
    /// Index in node list.
    pub idx: usize,

    pub bbox: Vec<Range<f32>>,
    pub wait_points: Vec<Point<'v>>,
    pub delay_points: Vec<Point<'v>>,
    pub max_point: Cell<Point<'v>>,
    pub left: Option<Rc<RefCell<KDNode<'v>>>>,
    pub right: Option<Rc<RefCell<KDNode<'v>>>>,
}

impl<'v> Default for KDNode<'v> {
    fn default() -> Self {
        Self {
            points: Rc::new(RefCell::new(Vec::new())),
            point_left: 0,
            point_right: 0,
            idx: 0,
            bbox: Vec::new(),
            wait_points: Vec::new(),
            delay_points: Vec::new(),
            max_point: Cell::new(Point::default()),
            left: None,
            right: None,
        }
    }
}

impl<'v> KDNode<'v> {
    pub fn init(&mut self, ref_point: Point<'v>) {
        self.wait_points.clear();
        self.delay_points.clear();
        if self.left.is_some() && self.right.is_some() {
            let left_rc = self.left.clone().unwrap();
            let mut left = left_rc.borrow_mut();
            let right_rc = self.right.clone().unwrap();
            let mut right = right_rc.borrow_mut();
            left.init(ref_point.clone());
            right.init(ref_point.clone());

            let left_max_point = left.max_point.get_mut().clone();
            let right_max_point = right.max_point.get_mut().clone();
            self.update_max_point(left_max_point, right_max_point);
        } else {
            let max_dis = f32::NEG_INFINITY;
            let points_rc = self.points.clone();
            let points = points_rc.borrow_mut();
            for i in self.point_left..self.point_right {
                let dis = points[i].update_distance(&ref_point);
                if dis > max_dis {
                    self.max_point.set(points[i].clone())
                }
            }
        }
    }

    fn update_max_point(&self, lpoint: Point<'v>, rpoint: Point<'v>) {
        if lpoint.dis > rpoint.dis {
            self.max_point.set(lpoint);
        } else {
            self.max_point.set(rpoint);
        }
    }

    /// Return the **squared** dim distance from the reference point to the bounding box.
    fn bound_distance(&self, ref_point: Point<'v>) -> f32 {
        let ret = (0..self.bbox.len()).fold(0f32, |acc, x| {
            let dim_distance;
            if ref_point[x] < self.bbox[x].start {
                dim_distance = self.bbox[x].start - ref_point[x];
            } else if ref_point[x] > self.bbox[x].end {
                dim_distance = ref_point[x] - self.bbox[x].end;
            } else {
                dim_distance = 0f32;
            }
            acc + dim_distance.powi(2)
        });
        ret
    }

    pub fn send_delay_point(&mut self, ref_point: Point<'v>) {
        self.wait_points.push(ref_point);
    }

    pub fn update_distance(&mut self) {
        for ref_point in self.wait_points.iter() {
            let lastmax_distance = self.max_point.get_mut().dis.get();
            let cur_distance = self.max_point.get_mut().distance(ref_point);
            if cur_distance > lastmax_distance {
                let boundary_distance = self.bound_distance(ref_point.clone());
                if boundary_distance < lastmax_distance {
                    self.delay_points.push(ref_point.clone());
                }
            } else {
                if self.left.is_some() && self.right.is_some() {
                    let left_rc = self.left.clone().unwrap();
                    let mut left = left_rc.borrow_mut();
                    let right_rc = self.right.clone().unwrap();
                    let mut right = right_rc.borrow_mut();
                    if !self.delay_points.is_empty() {
                        for delay_point in self.delay_points.iter() {
                            left.send_delay_point(delay_point.clone());
                            right.send_delay_point(delay_point.clone());
                        }
                        self.delay_points.clear();
                    }
                    left.send_delay_point(ref_point.clone());
                    left.update_distance();

                    right.send_delay_point(ref_point.clone());
                    right.update_distance();
                    let left_max_point = left.max_point.get_mut().clone();
                    let right_max_point = right.max_point.get_mut().clone();
                    self.update_max_point(left_max_point, right_max_point);
                } else {
                    self.delay_points.push(ref_point.clone());
                    let points_rc = self.points.clone();
                    let points = points_rc.borrow_mut();
                    for delay_point in self.delay_points.iter() {
                        let mut max_dis = f32::NEG_INFINITY;
                        for i in self.point_left..self.point_right {
                            let dis = points[i].update_distance(delay_point);
                            if dis > max_dis {
                                max_dis = dis;
                                self.max_point.set(points[i].clone());
                            }
                        }
                    }
                    self.delay_points.clear();
                }
            }
        }
        self.wait_points.clear();
    }

    pub fn size(&self) -> usize {
        if let (Some(left), Some(right)) = (&self.left, &self.right) {
            left.borrow_mut().size() + right.borrow_mut().size()
        } else {
            self.point_right - self.point_left
        }
    }
}
