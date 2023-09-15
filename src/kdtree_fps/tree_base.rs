use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;

use super::node::KDNode;
use super::point::Point;

// pub(super) struct KDTreeBase<'v> {
//     pub sampled_points: Vec<&'v Point<'v>>,
//     pub points: &'v [Point<'v>],
// }

// impl<'a> KDTreeBase<'a> {
//     pub fn point_size(&self) -> usize {
//         self.points.len()
//     }
// }

pub(super) trait KDTreeBase<'v> {
    fn get_points(&self) -> Rc<RefCell<Vec<Point<'v>>>>;

    fn get_root(&self) -> Rc<RefCell<KDNode<'v>>>;
    fn set_root(&mut self, root: Rc<RefCell<KDNode<'v>>>);

    fn add_sampled_point(&mut self, point: Point<'v>);

    // virtual methods
    fn add_node(&mut self, node: Rc<RefCell<KDNode<'v>>>);
    fn leaf_node(&self, high: usize, count: usize) -> bool;
    fn max_point(&self) -> Point<'v>;
    fn update_distance(&mut self, ref_point: Point<'v>);
    fn sample(self, n_sample: usize) -> Vec<Point<'v>>;

    // base methods

    fn init(&mut self, ref_point: Point<'v>) {
        self.add_sampled_point(ref_point.clone());
        let root = self.get_root();
        root.borrow_mut().init(ref_point);
    }

    fn build_tree(&mut self) {
        let (left, right) = (0, self.get_points().borrow().len());
        let bbox = self.compute_bbox(left, right);
        let node = self.divide_tree(left, right, &bbox, 0);
        self.set_root(node);
    }

    fn divide_tree(
        &mut self,
        left: usize,
        right: usize,
        bbox: &[Range<f32>],
        curr_high: usize,
    ) -> Rc<RefCell<KDNode<'v>>> {
        let mut node = KDNode::default();
        node.bbox = bbox.to_vec();

        let count = right - left;
        if self.leaf_node(curr_high, count) {
            node.point_left = left;
            node.point_right = right;
            node.points = self.get_points();
            let ret = Rc::new(RefCell::new(node));
            self.add_node(ret.clone());
            ret
        } else {
            let split_dim = self.find_split_dim(bbox);
            let split_val = self.select_median(split_dim, left, right);
            let split_delta = self.plane_split(left, right, split_dim, split_val);

            let bbox_cur = self.compute_bbox(left, left + split_delta);
            node.left = Some(self.divide_tree(left, left + split_delta, &bbox_cur, curr_high + 1));
            let bbox_cur = self.compute_bbox(left + split_delta, right);
            node.right =
                Some(self.divide_tree(left + split_delta, right, &bbox_cur, curr_high + 1));
            Rc::new(RefCell::new(node))
        }
    }

    fn plane_split(&self, left: usize, right: usize, split_dim: usize, split_val: f32) -> usize {
        let points_rc = self.get_points();
        let mut points_ = points_rc.borrow_mut();
        let mut start = left;
        let mut end = right - 1;

        loop {
            while start <= end && points_[start].pos[split_dim] < split_val {
                start += 1;
            } // find the first point that is not less than split_val
            while start <= end && points_[end].pos[split_dim] >= split_val {
                end -= 1;
            } // find the first point that is not greater than split_val
            if start > end {
                break;
            } else {
                points_.swap(start, end);
                start += 1;
                end -= 1;
            }
        }

        if start == left {
            1
        } else if start == right {
            right - left - 1
        } else {
            start - left
        }
    }

    fn select_median(&self, dim: usize, left: usize, right: usize) -> f32 {
        let points_rc = self.get_points();
        let points_ = points_rc.borrow();
        let sum = points_[left..right]
            .iter()
            .map(|x| x[dim])
            .fold(0f32, |acc, x| acc + x);
        let median = sum / (right - left) as f32;
        median
    }

    fn find_split_dim(&self, bbox: &[Range<f32>]) -> usize {
        let dim = bbox.len();
        let mut best_dim: usize = 0;
        let mut max_span = f32::NEG_INFINITY;

        for cur_dim in 0..dim {
            let span = bbox[cur_dim].end - bbox[cur_dim].start;
            if span > max_span {
                max_span = span;
                best_dim = cur_dim;
            }
        }

        best_dim
    }

    #[inline]
    fn compute_bbox(&self, left: usize, right: usize) -> Vec<Range<f32>> {
        let points_rc = self.get_points();
        let points_ = points_rc.borrow();
        let dim = points_[0].dim();

        let mut bbox = vec![
            Range::<f32> {
                start: f32::INFINITY,
                end: f32::NEG_INFINITY,
            };
            dim
        ];

        for i in left..right {
            for cur_dim in 0..dim {
                let val = points_[i][cur_dim];
                bbox[cur_dim].start = f32::min(bbox[cur_dim].start, val);
                bbox[cur_dim].end = f32::max(bbox[cur_dim].end, val);
            }
        }
        bbox
    }

    #[inline]
    fn compute_minmax(&self, left: usize, right: usize, dim: usize) -> Range<f32> {
        let points_rc = self.get_points();
        let points_ = points_rc.borrow();
        let mut ret = Range::<f32> {
            start: f32::INFINITY,
            end: f32::NEG_INFINITY,
        };
        for i in left..right {
            let val = points_[i][dim];
            ret.start = f32::min(ret.start, val);
            ret.end = f32::max(ret.end, val);
        }
        ret
    }
}
