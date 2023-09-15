use std::cell::RefCell;
use std::rc::Rc;

use super::node::KDNode;
use super::point::Point;
use super::tree_base::KDTreeBase;

pub struct LineTree<'v> {
    sampled_points: Vec<Point<'v>>,
    points: Rc<RefCell<Vec<Point<'v>>>>,
    root: Rc<RefCell<KDNode<'v>>>,
    node_list: Vec<Rc<RefCell<KDNode<'v>>>>,
    height: usize,
}

impl<'v> LineTree<'v> {
    pub fn new(points: Rc<RefCell<Vec<Point<'v>>>>, height: usize) -> Self {
        Self {
            sampled_points: Vec::new(),
            points,
            root: Rc::new(RefCell::new(KDNode::default())),
            node_list: Vec::new(),
            height,
        }
    }
}

impl<'v> KDTreeBase<'v> for LineTree<'v> {
    fn get_points(&self) -> Rc<RefCell<Vec<Point<'v>>>> {
        self.points.clone()
    }

    fn get_root(&self) -> Rc<RefCell<KDNode<'v>>> {
        self.root.clone()
    }

    fn set_root(&mut self, root: Rc<RefCell<KDNode<'v>>>) {
        self.root = root;
    }

    fn add_sampled_point(&mut self, point: Point<'v>) {
        self.sampled_points.push(point);
    }

    fn add_node(&mut self, node: Rc<RefCell<KDNode<'v>>>) {
        {
            let mut node_mut = node.borrow_mut();
            node_mut.idx = self.node_list.len();
        }
        self.node_list.push(node);
    }

    fn leaf_node(&self, high: usize, count: usize) -> bool {
        high == self.height || count == 1
    }

    fn max_point(&self) -> Point<'v> {
        let mut max_dist = f32::NEG_INFINITY;
        let mut ret_point = Point::default();
        for bucket in self.node_list.iter() {
            let mut bucket = bucket.borrow_mut();
            let max_point = bucket.max_point.get_mut();
            let dist = max_point.dis.get();
            if dist > max_dist {
                max_dist = dist;
                ret_point = max_point.clone();
            }
        }
        ret_point
    }

    fn update_distance(&mut self, ref_point: Point<'v>) {
        for bucket in self.node_list.iter() {
            let mut bucket = bucket.borrow_mut();
            bucket.send_delay_point(ref_point.clone());
            bucket.update_distance();
        }
    }

    fn sample(mut self, n_sample: usize) -> Vec<Point<'v>> {
        for _ in 0..n_sample {
            let ref_point = self.max_point();
            self.add_sampled_point(ref_point.clone());
            self.update_distance(ref_point);
        }
        self.sampled_points
    }
}

// test
#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::fpsample::init;

    use super::*;
    use numpy::ndarray;
    use rand::prelude::*;

    fn create_pos(n: usize) -> Vec<ndarray::Array1<f32>> {
        let mut rng = thread_rng();
        let mut ret = Vec::new();
        for _ in 0..n {
            let pos = ndarray::arr1(&[rng.gen(), rng.gen(), rng.gen()]);
            ret.push(pos);
        }
        ret
    }

    #[test]
    fn test_build_tree1() {
        let pos = ndarray::arr1(&[0.0, 0.0, 0.0]);
        let point = Point::new(pos.view(), None);
        let points = Rc::new(RefCell::new(vec![point]));
        let mut tree = LineTree::new(points, 1);
        tree.build_tree();
    }

    #[test]
    fn test_build_tree2() {
        let pos = create_pos(64);
        let points: Vec<Point> = pos.iter().map(|x| Point::new(x.view(), None)).collect();
        let shared_points = Rc::new(RefCell::new(points));
        let mut tree = LineTree::new(shared_points, 5);
        tree.build_tree();
    }

    #[test]
    fn test_sample1() {
        let pos = create_pos(50000);
        let points: Vec<Point> = pos.iter().map(|x| Point::new(x.view(), None)).collect();
        let init_point = points[0].clone();
        let shared_points = Rc::new(RefCell::new(points));
        let mut tree = LineTree::new(shared_points, 8);
        let start_time = Instant::now();
        tree.build_tree();
        tree.init(init_point);
        let _sampled_points = tree.sample(4096);
        let time_cost = start_time.elapsed().as_millis();
        println!("time cost: {} ms", time_cost);
    }
}
